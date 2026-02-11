"""
Tests for reduction operations using PyPTO frontend.

Tests tile-level reduction operations:
- row_max: Row-wise maximum reduction
- row_sum: Row-wise sum reduction

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
# Row-wise max reduction
# =============================================================================


class TestTileRowMax(PTOTestCase):
    """Base class for tile row-wise max reduction tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_row_max_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec("c", [self.ROWS, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.max(tensors["a"], axis=1, keepdims=True)


class TestTileRowMax64x64(TestTileRowMax):
    """64x64 tile row-wise max reduction test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowMaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_max(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tmp_tile: pl.Tile[[64, 64], pl.FP32] = pl.op.create_tile(
                    [64, 64], dtype=pl.FP32, target_memory=1
                )
                tile_c = pl.op.block.row_max(tile_a, tmp_tile)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 1], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
                out_c = self.tile_row_max(a)
                return out_c

        return TileRowMaxProgram


class TestTileRowMax128x128(TestTileRowMax):
    """128x128 tile row-wise max reduction test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowMaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_max(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tmp_tile: pl.Tile[[128, 128], pl.FP32] = pl.op.create_tile(
                    [128, 128], dtype=pl.FP32, target_memory=1
                )
                tile_c = pl.op.block.row_max(tile_a, tmp_tile)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 1], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 1], pl.FP32]:
                out_c = self.tile_row_max(a)
                return out_c

        return TileRowMaxProgram


# =============================================================================
# Row-wise sum reduction
# =============================================================================


class TestTileRowSum(PTOTestCase):
    """Base class for tile row-wise sum reduction tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_row_sum_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec("c", [self.ROWS, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.sum(tensors["a"], axis=1, keepdims=True)


class TestTileRowSum64x64(TestTileRowSum):
    """64x64 tile row-wise sum reduction test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowSumProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_sum(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tmp_tile: pl.Tile[[64, 64], pl.FP32] = pl.op.create_tile(
                    [64, 64], dtype=pl.FP32, target_memory=1
                )
                tile_c = pl.op.block.row_sum(tile_a, tmp_tile)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 1], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
                out_c = self.tile_row_sum(a)
                return out_c

        return TileRowSumProgram


class TestTileRowSum128x128(TestTileRowSum):
    """128x128 tile row-wise sum reduction test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileRowSumProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_row_sum(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tmp_tile: pl.Tile[[128, 128], pl.FP32] = pl.op.create_tile(
                    [128, 128], dtype=pl.FP32, target_memory=1
                )
                tile_c = pl.op.block.row_sum(tile_a, tmp_tile)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 1], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 1], pl.FP32]:
                out_c = self.tile_row_sum(a)
                return out_c

        return TileRowSumProgram


# =============================================================================
# pytest test functions
# =============================================================================


class TestReductionOperations:
    """Test suite for reduction operations."""

    # Row-wise max reduction
    def test_tile_row_max_64x64(self, test_runner):
        """Test tile row-wise max reduction with 64x64 shape."""
        test_case = TestTileRowMax64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_row_max_128x128(self, test_runner):
        """Test tile row-wise max reduction with 128x128 shape."""
        test_case = TestTileRowMax128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Row-wise sum reduction
    def test_tile_row_sum_64x64(self, test_runner):
        """Test tile row-wise sum reduction with 64x64 shape."""
        test_case = TestTileRowSum64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_row_sum_128x128(self, test_runner):
        """Test tile row-wise sum reduction with 128x128 shape."""
        test_case = TestTileRowSum128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"
