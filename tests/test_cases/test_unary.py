"""
Tests for unary operations using PyPTO frontend.

Tests tile-level unary operations:
- log: Natural logarithm
- abs: Absolute value
- relu: ReLU activation (max(0, x))
- exp: Exponential
- sqrt: Square root
- neg: Negation

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
# Natural logarithm
# =============================================================================


class TestTileLog(PTOTestCase):
    """Base class for tile natural logarithm tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_log_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=2.718),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.log(tensors["a"])


class TestTileLog64x64(TestTileLog):
    """64x64 tile natural logarithm test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileLogProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_log(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.log(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_log(a)
                return out_c

        return TileLogProgram


class TestTileLog128x128(TestTileLog):
    """128x128 tile natural logarithm test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileLogProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_log(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.log(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_log(a)
                return out_c

        return TileLogProgram


# =============================================================================
# Absolute value
# =============================================================================


class TestTileAbs(PTOTestCase):
    """Base class for tile absolute value tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_abs_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape) * 2 - 1,
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.abs(tensors["a"])


class TestTileAbs64x64(TestTileAbs):
    """64x64 tile absolute value test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileAbsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_abs(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.abs(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_abs(a)
                return out_c

        return TileAbsProgram


class TestTileAbs128x128(TestTileAbs):
    """128x128 tile absolute value test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileAbsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_abs(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.abs(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_abs(a)
                return out_c

        return TileAbsProgram


# =============================================================================
# ReLU activation
# =============================================================================


class TestTileRelu(PTOTestCase):
    """Base class for tile ReLU activation tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_relu_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape) * 2,
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.maximum(0, tensors["a"])


class TestTileRelu64x64(TestTileRelu):
    """64x64 tile ReLU activation test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileReluProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_relu(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.relu(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_relu(a)
                return out_c

        return TileReluProgram


class TestTileRelu128x128(TestTileRelu):
    """128x128 tile ReLU activation test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileReluProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_relu(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.relu(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_relu(a)
                return out_c

        return TileReluProgram


# =============================================================================
# Exponential
# =============================================================================


class TestTileExp(PTOTestCase):
    """Base class for tile exponential tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_exp_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=1.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.exp(tensors["a"])


class TestTileExp64x64(TestTileExp):
    """64x64 tile exponential test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileExpProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_exp(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.exp(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_exp(a)
                return out_c

        return TileExpProgram


class TestTileExp128x128(TestTileExp):
    """128x128 tile exponential test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileExpProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_exp(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.exp(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_exp(a)
                return out_c

        return TileExpProgram


# =============================================================================
# Square root
# =============================================================================


class TestTileSqrt(PTOTestCase):
    """Base class for tile square root tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_sqrt_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=4.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.sqrt(tensors["a"])


class TestTileSqrt64x64(TestTileSqrt):
    """64x64 tile square root test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileSqrtProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_sqrt(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.sqrt(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_sqrt(a)
                return out_c

        return TileSqrtProgram


class TestTileSqrt128x128(TestTileSqrt):
    """128x128 tile square root test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileSqrtProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_sqrt(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.sqrt(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_sqrt(a)
                return out_c

        return TileSqrtProgram


# =============================================================================
# Negation
# =============================================================================


class TestTileNeg(PTOTestCase):
    """Base class for tile negation tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_neg_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=3.5),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = -tensors["a"]


class TestTileNeg64x64(TestTileNeg):
    """64x64 tile negation test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileNegProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_neg(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.neg(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_neg(a)
                return out_c

        return TileNegProgram


class TestTileNeg128x128(TestTileNeg):
    """128x128 tile negation test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileNegProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_neg(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.neg(tile_a)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_neg(a)
                return out_c

        return TileNegProgram


# =============================================================================
# pytest test functions
# =============================================================================


class TestUnaryOperations:
    """Test suite for unary operations."""

    # Natural logarithm
    def test_tile_log_64x64(self, test_runner):
        """Test tile natural logarithm with 64x64 shape."""
        test_case = TestTileLog64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_log_128x128(self, test_runner):
        """Test tile natural logarithm with 128x128 shape."""
        test_case = TestTileLog128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Absolute value
    def test_tile_abs_64x64(self, test_runner):
        """Test tile absolute value with 64x64 shape."""
        test_case = TestTileAbs64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_abs_128x128(self, test_runner):
        """Test tile absolute value with 128x128 shape."""
        test_case = TestTileAbs128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # ReLU activation
    def test_tile_relu_64x64(self, test_runner):
        """Test tile ReLU activation with 64x64 shape."""
        test_case = TestTileRelu64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_relu_128x128(self, test_runner):
        """Test tile ReLU activation with 128x128 shape."""
        test_case = TestTileRelu128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Exponential
    def test_tile_exp_64x64(self, test_runner):
        """Test tile exponential with 64x64 shape."""
        test_case = TestTileExp64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_exp_128x128(self, test_runner):
        """Test tile exponential with 128x128 shape."""
        test_case = TestTileExp128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Square root
    def test_tile_sqrt_64x64(self, test_runner):
        """Test tile square root with 64x64 shape."""
        test_case = TestTileSqrt64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_sqrt_128x128(self, test_runner):
        """Test tile square root with 128x128 shape."""
        test_case = TestTileSqrt128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Negation
    def test_tile_neg_64x64(self, test_runner):
        """Test tile negation with 64x64 shape."""
        test_case = TestTileNeg64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_neg_128x128(self, test_runner):
        """Test tile negation with 128x128 shape."""
        test_case = TestTileNeg128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"
