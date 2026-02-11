"""
Tests for memory operations using PyPTO frontend.

Tests tile-level memory operations:
- load + store: Basic memory copy
- full: Create constant-filled tiles

These tests use the simplified pattern where orchestration is auto-generated.
Each operation has both 64x64 and 128x128 test cases.

Note: Operations like get_block_idx, create_tile, alloc, and move are auxiliary
operations that are already used in other tests (e.g., reduction tests use create_tile,
matmul tests use move for L0A/L0B transfers).
"""

import sys
from typing import Any, List

import numpy as np

from pto_test.core import environment
from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# Add pypto to path
_PYPTO_PYTHON = environment.get_pypto_python_path()
if _PYPTO_PYTHON is not None and _PYPTO_PYTHON.exists() and str(_PYPTO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYPTO_PYTHON))


# =============================================================================
# Load + Store: Basic memory copy
# =============================================================================


class TestTileLoadStore(PTOTestCase):
    """Base class for tile load + store tests (memory copy)."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_load_store_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        del params  # Unused
        tensors["c"][:] = tensors["a"]


class TestTileLoadStore64x64(TestTileLoadStore):
    """64x64 tile load + store test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileLoadStoreProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_load_store(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                out_c = pl.op.block.store(tile_a, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_load_store(a)
                return out_c

        return TileLoadStoreProgram


class TestTileLoadStore128x128(TestTileLoadStore):
    """128x128 tile load + store test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileLoadStoreProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_load_store(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                out_c = pl.op.block.store(tile_a, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_load_store(a)
                return out_c

        return TileLoadStoreProgram


# =============================================================================
# Full: Create constant-filled tiles
# =============================================================================


class TestTileFull(PTOTestCase):
    """Base class for tile full tests (constant initialization)."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses
    FILL_VALUE = 3.14  # Constant value to fill

    def get_name(self) -> str:
        return f"tile_full_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        del params  # Unused
        tensors["c"][:] = 3.14  # Constant fill value


class TestTileFull64x64(TestTileFull):
    """64x64 tile full test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileFullProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_full(
                self,
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_c = pl.op.block.full([64, 64], dtype=pl.FP32, value=3.14)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_full()
                return out_c

        return TileFullProgram


class TestTileFull128x128(TestTileFull):
    """128x128 tile full test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileFullProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_full(
                self,
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_c = pl.op.block.full([128, 128], dtype=pl.FP32, value=3.14)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_full()
                return out_c

        return TileFullProgram


# =============================================================================
# pytest test functions
# =============================================================================


class TestMemoryOperations:
    """Test suite for memory operations."""

    # Load + Store
    def test_tile_load_store_64x64(self, test_runner):
        """Test tile load + store with 64x64 shape."""
        test_case = TestTileLoadStore64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_load_store_128x128(self, test_runner):
        """Test tile load + store with 128x128 shape."""
        test_case = TestTileLoadStore128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Full
    def test_tile_full_64x64(self, test_runner):
        """Test tile full with 64x64 shape."""
        test_case = TestTileFull64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_full_128x128(self, test_runner):
        """Test tile full with 128x128 shape."""
        test_case = TestTileFull128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"
