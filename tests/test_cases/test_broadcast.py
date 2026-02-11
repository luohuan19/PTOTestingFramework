"""
Tests for broadcast operations using PyPTO frontend.

Tests tile-level broadcast operations:
- row_expand_sub: Row-wise broadcast subtraction
- row_expand_mul: Row-wise broadcast multiplication
- row_expand_div: Row-wise broadcast division

These tests use the simplified pattern where orchestration is auto-generated.
Each operation has both 64x64 and 128x128 test cases.
"""

import sys
from pathlib import Path
from typing import Any, List

import numpy as np

from pto_test.core import environment
from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# Add pypto to path
_PYPTO_PYTHON = environment.get_pypto_python_path()
if _PYPTO_PYTHON is not None and _PYPTO_PYTHON.exists() and str(_PYPTO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYPTO_PYTHON))


# =============================================================================
# Row-wise broadcast subtraction
# =============================================================================


class TestTileRowExpandSub(PTOTestCase):
    """Base class for tile row-wise broadcast subtraction tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_row_expand_sub_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec(
                "row_vec",
                [self.ROWS, 1],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["row_vec"]


class TestTileRowExpandSub64x64(TestTileRowExpandSub):
    """64x64 tile row-wise broadcast subtraction test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowExpandSubProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_expand_sub(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                row_vec: pl.Tensor[[64, 1], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_row = pl.op.block.load(row_vec, [0, 0], [64, 1])
                tile_c = pl.op.block.row_expand_sub(tile_a, tile_row)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], row_vec: pl.Tensor[[64, 1], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_row_expand_sub(a, row_vec)
                return out_c

        return TileRowExpandSubProgram


class TestTileRowExpandSub128x128(TestTileRowExpandSub):
    """128x128 tile row-wise broadcast subtraction test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowExpandSubProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_expand_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_row = pl.op.block.load(row_vec, [0, 0], [128, 1])
                tile_c = pl.op.block.row_expand_sub(tile_a, tile_row)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], row_vec: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_sub(a, row_vec)
                return out_c

        return TileRowExpandSubProgram


# =============================================================================
# Row-wise broadcast multiplication
# =============================================================================


class TestTileRowExpandMul(PTOTestCase):
    """Base class for tile row-wise broadcast multiplication tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_row_expand_mul_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec(
                "row_vec",
                [self.ROWS, 1],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["row_vec"]


class TestTileRowExpandMul64x64(TestTileRowExpandMul):
    """64x64 tile row-wise broadcast multiplication test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowExpandMulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_expand_mul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                row_vec: pl.Tensor[[64, 1], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_row = pl.op.block.load(row_vec, [0, 0], [64, 1])
                tile_c = pl.op.block.row_expand_mul(tile_a, tile_row)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], row_vec: pl.Tensor[[64, 1], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_row_expand_mul(a, row_vec)
                return out_c

        return TileRowExpandMulProgram


class TestTileRowExpandMul128x128(TestTileRowExpandMul):
    """128x128 tile row-wise broadcast multiplication test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowExpandMulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_expand_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_row = pl.op.block.load(row_vec, [0, 0], [128, 1])
                tile_c = pl.op.block.row_expand_mul(tile_a, tile_row)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], row_vec: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_mul(a, row_vec)
                return out_c

        return TileRowExpandMulProgram


# =============================================================================
# Row-wise broadcast division
# =============================================================================


class TestTileRowExpandDiv(PTOTestCase):
    """Base class for tile row-wise broadcast division tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_row_expand_div_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec(
                "row_vec",
                [self.ROWS, 1],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape) + 1.0,  # Avoid division by zero
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["row_vec"]


class TestTileRowExpandDiv64x64(TestTileRowExpandDiv):
    """64x64 tile row-wise broadcast division test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowExpandDivProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_expand_div(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                row_vec: pl.Tensor[[64, 1], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_row = pl.op.block.load(row_vec, [0, 0], [64, 1])
                tile_c = pl.op.block.row_expand_div(tile_a, tile_row)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], row_vec: pl.Tensor[[64, 1], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_row_expand_div(a, row_vec)
                return out_c

        return TileRowExpandDivProgram


class TestTileRowExpandDiv128x128(TestTileRowExpandDiv):
    """128x128 tile row-wise broadcast division test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowExpandDivProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_expand_div(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_row = pl.op.block.load(row_vec, [0, 0], [128, 1])
                tile_c = pl.op.block.row_expand_div(tile_a, tile_row)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], row_vec: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_div(a, row_vec)
                return out_c

        return TileRowExpandDivProgram


# =============================================================================
# pytest test functions
# =============================================================================


class TestBroadcastOperations:
    """Test suite for broadcast operations."""

    # Row-wise broadcast subtraction
    def test_tile_row_expand_sub_64x64(self, test_runner):
        """Test tile row-wise broadcast subtraction with 64x64 shape."""
        test_case = TestTileRowExpandSub64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_row_expand_sub_128x128(self, test_runner):
        """Test tile row-wise broadcast subtraction with 128x128 shape."""
        test_case = TestTileRowExpandSub128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Row-wise broadcast multiplication
    def test_tile_row_expand_mul_64x64(self, test_runner):
        """Test tile row-wise broadcast multiplication with 64x64 shape."""
        test_case = TestTileRowExpandMul64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_row_expand_mul_128x128(self, test_runner):
        """Test tile row-wise broadcast multiplication with 128x128 shape."""
        test_case = TestTileRowExpandMul128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Row-wise broadcast division
    def test_tile_row_expand_div_64x64(self, test_runner):
        """Test tile row-wise broadcast division with 64x64 shape."""
        test_case = TestTileRowExpandDiv64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_row_expand_div_128x128(self, test_runner):
        """Test tile row-wise broadcast division with 128x128 shape."""
        test_case = TestTileRowExpandDiv128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"
