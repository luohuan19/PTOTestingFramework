"""
Tests for Paged Attention implementation using PyPTO frontend.

This test validates the paged attention kernels through the
pto-testing-framework, ensuring correct code generation and execution.

Paged Attention Algorithm:
  For each block j:
    1. QK matmul:  sij = qi @ kj.T          (AIC/Cube unit)
    2. Softmax:    pij = softmax(sij)       (AIV/Vector unit)
    3. PV matmul:  oi_new = pij @ vj        (AIC/Cube unit)
    4. Update:     oi = online_update(...)  (AIV/Vector unit)

Dimension semantics:
  - num_heads: number of attention heads
  - head_dim: dimension of each head
  - block_size: KV cache block size

Kernels tested (4 kernels):
  - qk_matmul: Q @ K^T matrix multiplication (AIC/Cube unit)
  - softmax_prepare: scale -> rowmax -> exp -> rowsum (AIV/Vector unit)
  - pv_matmul: P @ V matrix multiplication (AIC/Cube unit)
  - online_update: online softmax accumulation + normalize (AIV/Vector unit)
"""

from typing import Any, List

import numpy as np
import pytest

from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec


# Scale factor: 1/sqrt(head_dim), default for head_dim=128
DEFAULT_SCALE = 0.0884

# Number of blocks for online update test
NUM_BLOCKS = 2


class QKMatmulTestCase(PTOTestCase):
    """Test case for QK matmul kernel: sij = qi @ kj.T
    
    Computes attention scores between query and key vectors.
    - qi: Query vector, shape (num_heads, head_dim)
    - kj_t: Key vectors transposed, shape (head_dim, block_size)
    - sij: Attention scores, shape (num_heads, block_size)
    
    Note: In simplified 16x16 test, num_heads = head_dim = block_size = 16
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"qk_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("qi", [self.num_heads, self.head_dim], DataType.FP32, init_value=2.0),
            TensorSpec("kj_t", [self.head_dim, self.num_heads], DataType.FP32, init_value=3.0),
            TensorSpec("sij", [self.num_heads, self.num_heads], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi: pl.Tensor[[16, 16], pl.FP32],
                kj_t: pl.Tensor[[16, 16], pl.FP32],
                sij: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                qi_l1 = pl.load(qi, [0, 0], [16, 16], target_memory=2)
                kj_l1 = pl.load(kj_t, [0, 0], [16, 16], target_memory=2)
                qi_l0a = pl.move(qi_l1, target_memory=3)
                kj_l0b = pl.move(kj_l1, target_memory=4)
                sij_l0c = pl.matmul(qi_l0a, kj_l0b)
                out_sij = pl.l0c_store(sij_l0c, [0, 0], [16, 16], sij)
                return out_sij

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, qi: pl.Tensor[[16, 16], pl.FP32], kj_t: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_sij = self.qk_matmul(qi, kj_t)
                return out_sij

        return QKMatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["sij"][:] = np.matmul(tensors["qi"], tensors["kj_t"])


class SoftmaxPrepareTestCase(PTOTestCase):
    """Test case for softmax_prepare kernel: pij, mij, lij = softmax(sij * scale)
    
    Computes softmax probabilities with row-wise max and sum for online update.
    - sij: Attention scores, shape (num_heads, block_size)
    - pij: Softmax probabilities (unnormalized exp), shape (num_heads, block_size)
    - mij: Row-wise max values, shape (num_heads, 1)
    - lij: Row-wise sum of exp values, shape (num_heads, 1)
    
    Algorithm:
      sij_scaled = sij * scale
      mij = row_max(sij_scaled)
      pij = exp(sij_scaled - mij)  # row broadcast subtraction
      lij = row_sum(pij)
    """

    def __init__(self, num_heads: int = 16, block_size: int = 16, scale: float = DEFAULT_SCALE, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.block_size = block_size
        self.scale = scale

    def get_name(self) -> str:
        return f"softmax_prepare_{self.num_heads}h_{self.block_size}b"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("sij", [self.num_heads, self.block_size], DataType.FP32, init_value=1.0),
            TensorSpec("pij", [self.num_heads, self.block_size], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class SoftmaxPrepareProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def softmax_prepare(
                self,
                sij: pl.Tensor[[16, 16], pl.FP32],
                pij: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # Load sij to UB (target_memory=1)
                sij_tile = pl.load(sij, [0, 0], [16, 16], target_memory=1)
                # Scale: sij * scale_factor
                sij_scaled = pl.muls(sij_tile, 0.5)
                # Create temp tile for row reduction
                tmp_tile = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=1)
                # Row max: mij = max(sij_scaled, axis=1)
                mij_tile = pl.row_max(sij_scaled, tmp_tile)
                # Row broadcast subtraction: sij_scaled - mij
                sij_centered = pl.row_expand_sub(sij_scaled, mij_tile)
                # Exp: exp(sij_centered)
                pij_tile = pl.exp(sij_centered)
                # Row sum: lij = sum(pij, axis=1)
                # lij_tile = pl.row_sum(pij_tile, tmp_tile)
                # Store results
                pij_out = pl.store(pij_tile, [0, 0], [16, 16], pij)
                return pij_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, sij: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pij_o = self.softmax_prepare(sij)
                return pij_o

        return SoftmaxPrepareProgram

    def compute_expected(self, tensors, params=None):
        sij = tensors["sij"]
        sij_scaled = sij * 0.5
        mij = np.max(sij_scaled, axis=1, keepdims=True)
        pij = np.exp(sij_scaled - mij)
        tensors["pij"][:] = pij


class PVMatmulTestCase(PTOTestCase):
    """Test case for PV matmul kernel: oi_new = pij @ vj
    
    Computes weighted sum of value vectors.
    - pij: Softmax probabilities, shape (num_heads, block_size)
    - vj: Value vectors, shape (block_size, head_dim)
    - oi_new: Output for current block, shape (num_heads, head_dim)
    
    Note: In simplified 16x16 test, num_heads = head_dim = block_size = 16
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"pv_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("pij", [self.num_heads, self.num_heads], DataType.FP32, init_value=0.1),
            TensorSpec("vj", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.5),
            TensorSpec("oi_new", [self.num_heads, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class PVMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def pv_matmul(
                self,
                pij: pl.Tensor[[16, 16], pl.FP32],
                vj: pl.Tensor[[16, 16], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pij_l1 = pl.load(pij, [0, 0], [16, 16], target_memory=2)
                vj_l1 = pl.load(vj, [0, 0], [16, 16], target_memory=2)
                pij_l0a = pl.move(pij_l1, target_memory=3)
                vj_l0b = pl.move(vj_l1, target_memory=4)
                oi_l0c = pl.matmul(pij_l0a, vj_l0b)
                out_oi = pl.l0c_store(oi_l0c, [0, 0], [16, 16], oi_new)
                return out_oi

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, pij: pl.Tensor[[16, 16], pl.FP32], vj: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_oi = self.pv_matmul(pij, vj)
                return out_oi

        return PVMatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["oi_new"][:] = np.matmul(tensors["pij"], tensors["vj"])


class OnlineUpdateTestCase(PTOTestCase):
    """Test case for online_update kernel: online softmax accumulation.
    
    Updates the running attention output using online softmax algorithm.
    
    Inputs:
      - mi: Previous max values, shape (num_heads, 1)
      - mij: Current block max values, shape (num_heads, 1)
      - li: Previous sum values, shape (num_heads, 1)
      - lij: Current block sum values, shape (num_heads, 1)
      - oi: Previous output, shape (num_heads, head_dim)
      - oi_new: Current block output, shape (num_heads, head_dim)
    
    Outputs:
      - oi_updated: Updated output (before final normalization)
    
    Algorithm:
      mi_new = max(mi, mij)
      alpha = exp(mi - mi_new)
      beta = exp(mij - mi_new)
      li_updated = alpha * li + beta * lij
      oi_updated = alpha * oi + beta * oi_new
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"online_update_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("mij", [self.num_heads, 1], DataType.FP32, init_value=0.5),
            TensorSpec("lij", [self.num_heads, 1], DataType.FP32, init_value=1.5),
            TensorSpec("oi_new", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.3),
            TensorSpec("mi", [self.num_heads, 1], DataType.FP32, init_value=0.4),
            TensorSpec("li", [self.num_heads, 1], DataType.FP32, init_value=2.0),
            TensorSpec("oi", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.2),
            TensorSpec("dst", [self.num_heads, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class OnlineUpdateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 16], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # Load row vectors [16, 1] to UB
                mi_tile = pl.load(mi, [0, 0], [16, 1], target_memory=1)
                mij_tile = pl.load(mij, [0, 0], [16, 1], target_memory=1)
                li_tile = pl.load(li, [0, 0], [16, 1], target_memory=1)
                lij_tile = pl.load(lij, [0, 0], [16, 1], target_memory=1)
                # Load matrices [16, 16] to UB
                oi_tile = pl.load(oi, [0, 0], [16, 16], target_memory=1)
                oi_new_tile = pl.load(oi_new, [0, 0], [16, 16], target_memory=1)
                
                # dn to nd, reshape is inplace in isa, 
                # need close memory reuse pass to aviod conflict
                mi_tile_nd = pl.reshape(mi_tile, [1, 16])
                mij_tile_nd = pl.reshape(mij_tile, [1, 16])
                li_tile_nd = pl.reshape(li_tile, [1, 16])
                lij_tile_nd = pl.reshape(lij_tile, [1, 16])

                # mi_new = max(mi, mij)
                mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                
                # alpha = exp(mi - mi_new)
                mi_diff = pl.sub(mi_tile_nd, mi_new)
                alpha = pl.exp(mi_diff)
                
                # beta = exp(mij - mi_new)
                mij_diff = pl.sub(mij_tile_nd, mi_new)
                beta = pl.exp(mij_diff)
                
                # li_scaled = alpha * li
                li_scaled = pl.mul(alpha, li_tile_nd)
                
                # lij_scaled = beta * lij
                lij_scaled = pl.mul(beta, lij_tile_nd)
                
                # li_updated = li_scaled + lij_scaled
                li_updated = pl.add(li_scaled, lij_scaled)
                
                # oi_scaled = row_expand_mul(oi, alpha)
                alpha_dn = pl.reshape(alpha, [16, 1])
                oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                
                # oi_new_scaled = row_expand_mul(oi_new, beta)
                beta_dn = pl.reshape(beta, [16, 1])
                oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                
                # oi_updated = oi_scaled + oi_new_scaled
                oi_updated = pl.add(oi_scaled, oi_new_scaled)
                
                # dst = row_expand_div(oi_updated, li_updated)
                li_updated_dn = pl.reshape(li_updated, [16, 1])
                dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                
                # Store results
                dst_result = pl.store(dst_tile, [0, 0], [16, 16], dst)
                return dst_result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out = self.online_update(mij, lij, oi_new, mi, li, oi)
                return out

        return OnlineUpdateProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected output using online softmax update."""
        mi = tensors["mi"]
        mij = tensors["mij"]
        li = tensors["li"]
        lij = tensors["lij"]
        oi = tensors["oi"]
        oi_new = tensors["oi_new"]

        
        # mi_new = max(mi, mij) = 0.5
        mi_new = np.maximum(mi, mij)
        
        # alpha = exp(mi - mi_new), beta = exp(mij - mi_new)
        alpha = np.exp(mi - mi_new)  # = 0.9048
        beta = np.exp(mij - mi_new)  # = 1
        
        # li_updated = alpha * li + beta * lij
        li_updated = alpha * li + beta * lij    # = 3.3097
        
        # oi_scaled = oi * alpha (broadcast), oi_new_scaled = oi_new * beta (broadcast)
        oi_scaled = oi * alpha      
        oi_new_scaled = oi_new * beta
        
        # oi_updated = oi_scaled + oi_new_scaled
        oi_updated = oi_scaled + oi_new_scaled  # = 0.481
        
        # dst = oi_updated / li_updated (broadcast)  = 0.1453
        dst = oi_updated / li_updated  
        
        tensors["dst"][:] = dst


class OnlineUpdateMultiOutTestCase(PTOTestCase):
    """Test case for online_update kernel: online softmax accumulation.
    
    Updates the running attention output using online softmax algorithm.
    
    Inputs:
      - mi: Previous max values, shape (num_heads, 1)
      - mij: Current block max values, shape (num_heads, 1)
      - li: Previous sum values, shape (num_heads, 1)
      - lij: Current block sum values, shape (num_heads, 1)
      - oi: Previous output, shape (num_heads, head_dim)
      - oi_new: Current block output, shape (num_heads, head_dim)
    
    Outputs:
      - oi_updated: Updated output (before final normalization)
    
    Algorithm:
      mi_new = max(mi, mij)
      alpha = exp(mi - mi_new)
      beta = exp(mij - mi_new)
      li_updated = alpha * li + beta * lij
      oi_updated = alpha * oi + beta * oi_new
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"online_update_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("mij", [self.num_heads, 1], DataType.FP32, init_value=0.5),
            TensorSpec("lij", [self.num_heads, 1], DataType.FP32, init_value=1.5),
            TensorSpec("oi_new", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.3),
            TensorSpec("mi", [self.num_heads, 1], DataType.FP32, init_value=0.4, is_output=True),
            TensorSpec("li", [self.num_heads, 1], DataType.FP32, init_value=2.0, is_output=True),
            TensorSpec("oi", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.2, is_output=True),
            TensorSpec("dst", [self.num_heads, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class OnlineUpdateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 16], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32], 
                       pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                # Load row vectors [16, 1] to UB
                mi_tile = pl.load(mi, [0, 0], [16, 1], target_memory=1)
                mij_tile = pl.load(mij, [0, 0], [16, 1], target_memory=1)
                li_tile = pl.load(li, [0, 0], [16, 1], target_memory=1)
                lij_tile = pl.load(lij, [0, 0], [16, 1], target_memory=1)
                # Load matrices [16, 16] to UB
                oi_tile = pl.load(oi, [0, 0], [16, 16], target_memory=1)
                oi_new_tile = pl.load(oi_new, [0, 0], [16, 16], target_memory=1)
                
                # dn to nd, reshape is inplace in isa, 
                # need close memory reuse pass to aviod conflict
                mi_tile_nd = pl.reshape(mi_tile, [1, 16])
                mij_tile_nd = pl.reshape(mij_tile, [1, 16])
                li_tile_nd = pl.reshape(li_tile, [1, 16])
                lij_tile_nd = pl.reshape(lij_tile, [1, 16])

                # mi_new = max(mi, mij)
                mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                
                # alpha = exp(mi - mi_new)
                mi_diff = pl.sub(mi_tile_nd, mi_new)
                alpha = pl.exp(mi_diff)
                
                # beta = exp(mij - mi_new)
                mij_diff = pl.sub(mij_tile_nd, mi_new)
                beta = pl.exp(mij_diff)
                
                # li_scaled = alpha * li
                li_scaled = pl.mul(alpha, li_tile_nd)
                
                # lij_scaled = beta * lij
                lij_scaled = pl.mul(beta, lij_tile_nd)
                
                # li_updated = li_scaled + lij_scaled
                li_updated = pl.add(li_scaled, lij_scaled)
                
                # oi_scaled = row_expand_mul(oi, alpha)
                alpha_dn = pl.reshape(alpha, [16, 1])
                oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                
                # oi_new_scaled = row_expand_mul(oi_new, beta)
                beta_dn = pl.reshape(beta, [16, 1])
                oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                
                # oi_updated = oi_scaled + oi_new_scaled
                oi_updated = pl.add(oi_scaled, oi_new_scaled)
                
                # dst = row_expand_div(oi_updated, li_updated)
                li_updated_dn = pl.reshape(li_updated, [16, 1])
                dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                
                # Store results
                mi_new_dn = pl.reshape(mi_new, [16, 1])
                mi_result = pl.store(mi_new_dn, [0, 0], [16, 1], mi)
                li_result = pl.store(li_updated_dn, [0, 0], [16, 1], li)
                oi_result = pl.store(oi_updated, [0, 0], [16, 16], oi)
                dst_result = pl.store(dst_tile, [0, 0], [16, 16], dst)
                return mi_result, li_result, oi_result, dst_result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32], 
                       pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                mi, li, oi, dst = self.online_update(mij, lij, oi_new)
                return mi, li, oi, dst

        return OnlineUpdateProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected output using online softmax update."""
        mi = tensors["mi"]
        mij = tensors["mij"]
        li = tensors["li"]
        lij = tensors["lij"]
        oi = tensors["oi"]
        oi_new = tensors["oi_new"]
        
        # mi_new = max(mi, mij) = 0.5
        mi_new = np.maximum(mi, mij)
        
        # alpha = exp(mi - mi_new), beta = exp(mij - mi_new)
        alpha = np.exp(mi - mi_new)  # = 0.9048
        beta = np.exp(mij - mi_new)  # = 1
        
        # li_updated = alpha * li + beta * lij
        li_updated = alpha * li + beta * lij    # = 3.3097
        
        # oi_scaled = oi * alpha (broadcast), oi_new_scaled = oi_new * beta (broadcast)
        oi_scaled = oi * alpha      
        oi_new_scaled = oi_new * beta
        
        # oi_updated = oi_scaled + oi_new_scaled
        oi_updated = oi_scaled + oi_new_scaled  # = 0.481
        
        # dst = oi_updated / li_updated (broadcast)  = 0.1453
        dst = oi_updated / li_updated  
        
        tensors["mi"][:] = mi_new
        tensors["li"][:] = li_updated
        tensors["oi"][:] = oi_updated
        tensors["dst"][:] = dst


class TestPagedAttentionKernels:
    """Test suite for Paged Attention kernels (4 kernels).
    
    Parameters:
      - num_heads: number of attention heads
      - head_dim: dimension of each head
      - block_size: KV cache block size (for softmax)
    """

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_qk_matmul(self, test_runner, num_heads, head_dim):
        """Test QK matmul kernel: sij = qi @ kj^T"""
        test_case = QKMatmulTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"QK matmul test failed for {num_heads}h_{head_dim}d: {result.error}"

    @pytest.mark.parametrize("num_heads,block_size", [(16, 16)])
    def test_softmax_prepare(self, test_runner, num_heads, block_size):
        """Test softmax_prepare kernel: pij, mij, lij = softmax(sij * scale)"""
        test_case = SoftmaxPrepareTestCase(num_heads=num_heads, block_size=block_size)
        result = test_runner.run(test_case)
        assert result.passed, f"Softmax prepare test failed for {num_heads}h_{block_size}b: {result.error}"

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_pv_matmul(self, test_runner, num_heads, head_dim):
        """Test PV matmul kernel: oi_new = pij @ vj"""
        test_case = PVMatmulTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"PV matmul test failed for {num_heads}h_{head_dim}d: {result.error}"

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_online_update(self, test_runner, num_heads, head_dim):
        """Test online_update kernel: dst = (alpha*oi + beta*oi_new) / li_updated"""
        test_case = OnlineUpdateTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"Online update test failed for {num_heads}h_{head_dim}d: {result.error}"

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_multi_output(self, test_runner, num_heads, head_dim):
        """Test online_update kernel: dst = (alpha*oi + beta*oi_new) / li_updated"""
        test_case = OnlineUpdateMultiOutTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"Online update test failed for {num_heads}h_{head_dim}d: {result.error}"
