"""
Tests for elementwise operations using PyPTO frontend.

Tests tile-level operations including:
- Binary tile-tile operations: add, sub, mul, div, maximum
- Scalar operations: adds, subs, muls, divs

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


class TestTileAdd(PTOTestCase):
    """Base class for tile element-wise addition tests.

    Note: PyPTO requires shape dimensions to be compile-time constants in type
    annotations. Each shape needs its own test class with hardcoded dimensions.
    This is a limitation of PyPTO's type system.
    """

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_add_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self.ROWS, self.COLS], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        # PyPTO requires compile-time constant shapes in type annotations.
        # Subclasses must override this method with their specific shape.
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestTileAdd64x64(TestTileAdd):
    """64x64 tile addition test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_b = pl.op.block.load(b, [0, 0], [64, 64])
                tile_c = pl.op.block.add(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_add(a, b)
                return out_c

        return TileAddProgram


class TestTileAdd128x128(TestTileAdd):
    """128x128 tile addition test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_b = pl.op.block.load(b, [0, 0], [128, 128])
                tile_c = pl.op.block.add(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_add(a, b)
                return out_c

        return TileAddProgram


class TestTileMul(PTOTestCase):
    """Base class for tile element-wise multiplication tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_mul_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            # Method 1: Use Callable to generate random data (different on each run)
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            # Method 2: Use scalar value (recommended - simple and serializable)
            TensorSpec("b", [self.ROWS, self.COLS], DataType.FP32, init_value=3.0),
            # For other methods, see TestCustomArrayInit class examples:
            # - Small arrays can use np.array([[...]])
            # - Identity matrix: np.eye(n)
            # - Diagonal matrix: np.diag([...])
            # Output tensor: automatically zero-initialized
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TestTileMul64x64(TestTileMul):
    """64x64 tile multiplication test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_mul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_b = pl.op.block.load(b, [0, 0], [64, 64])
                tile_c = pl.op.block.mul(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_mul(a, b)
                return out_c

        return TileMulProgram


class TestTileMul128x128(TestTileMul):
    """128x128 tile multiplication test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_b = pl.op.block.load(b, [0, 0], [128, 128])
                tile_c = pl.op.block.mul(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_mul(a, b)
                return out_c

        return TileMulProgram


class TestTileSub(PTOTestCase):
    """Base class for tile element-wise subtraction tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_sub_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=5.0),
            TensorSpec("b", [self.ROWS, self.COLS], DataType.FP32, init_value=2.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["b"]


class TestTileSub64x64(TestTileSub):
    """64x64 tile subtraction test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileSubProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_sub(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_b = pl.op.block.load(b, [0, 0], [64, 64])
                tile_c = pl.op.block.sub(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_sub(a, b)
                return out_c

        return TileSubProgram


class TestTileSub128x128(TestTileSub):
    """128x128 tile subtraction test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileSubProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_b = pl.op.block.load(b, [0, 0], [128, 128])
                tile_c = pl.op.block.sub(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_sub(a, b)
                return out_c

        return TileSubProgram


class TestTileDiv(PTOTestCase):
    """Base class for tile element-wise division tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_div_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=6.0),
            TensorSpec("b", [self.ROWS, self.COLS], DataType.FP32, init_value=2.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["b"]


class TestTileDiv64x64(TestTileDiv):
    """64x64 tile division test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileDivProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_div(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_b = pl.op.block.load(b, [0, 0], [64, 64])
                tile_c = pl.op.block.div(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_div(a, b)
                return out_c

        return TileDivProgram


class TestTileDiv128x128(TestTileDiv):
    """128x128 tile division test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileDivProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_div(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_b = pl.op.block.load(b, [0, 0], [128, 128])
                tile_c = pl.op.block.div(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_div(a, b)
                return out_c

        return TileDivProgram


class TestTileMaximum(PTOTestCase):
    """Base class for tile element-wise maximum tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_maximum_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec(
                "b",
                [self.ROWS, self.COLS],
                DataType.FP32,
                init_value=lambda shape: np.random.randn(*shape),
            ),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.maximum(tensors["a"], tensors["b"])


class TestTileMaximum64x64(TestTileMaximum):
    """64x64 tile maximum test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMaximumProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_maximum(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_b = pl.op.block.load(b, [0, 0], [64, 64])
                tile_c = pl.op.block.maximum(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_maximum(a, b)
                return out_c

        return TileMaximumProgram


class TestTileMaximum128x128(TestTileMaximum):
    """128x128 tile maximum test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMaximumProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_maximum(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_b = pl.op.block.load(b, [0, 0], [128, 128])
                tile_c = pl.op.block.maximum(tile_a, tile_b)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_maximum(a, b)
                return out_c

        return TileMaximumProgram


# =============================================================================
# Scalar operations
# =============================================================================


class TestTileAdds(PTOTestCase):
    """Base class for tile-scalar addition tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_adds_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + 2.0


class TestTileAdds64x64(TestTileAdds):
    """64x64 tile-scalar addition test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileAddsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_adds(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.adds(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_adds(a)
                return out_c

        return TileAddsProgram


class TestTileAdds128x128(TestTileAdds):
    """128x128 tile-scalar addition test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileAddsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_adds(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.adds(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_adds(a)
                return out_c

        return TileAddsProgram


class TestTileSubs(PTOTestCase):
    """Base class for tile-scalar subtraction tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_subs_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=5.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - 2.0


class TestTileSubs64x64(TestTileSubs):
    """64x64 tile-scalar subtraction test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileSubsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_subs(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.subs(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_subs(a)
                return out_c

        return TileSubsProgram


class TestTileSubs128x128(TestTileSubs):
    """128x128 tile-scalar subtraction test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileSubsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_subs(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.subs(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_subs(a)
                return out_c

        return TileSubsProgram


class TestTileMuls(PTOTestCase):
    """Base class for tile-scalar multiplication tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_muls_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * 2.0


class TestTileMuls64x64(TestTileMuls):
    """64x64 tile-scalar multiplication test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMulsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_muls(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.muls(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_muls(a)
                return out_c

        return TileMulsProgram


class TestTileMuls128x128(TestTileMuls):
    """128x128 tile-scalar multiplication test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMulsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_muls(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.muls(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_muls(a)
                return out_c

        return TileMulsProgram


class TestTileDivs(PTOTestCase):
    """Base class for tile-scalar division tests."""

    ROWS = 128  # Override in subclasses
    COLS = 128  # Override in subclasses

    def get_name(self) -> str:
        return f"tile_divs_{self.ROWS}x{self.COLS}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.ROWS, self.COLS], DataType.FP32, init_value=6.0),
            TensorSpec("c", [self.ROWS, self.COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_program() with their specific shape")

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / 2.0


class TestTileDivs64x64(TestTileDivs):
    """64x64 tile-scalar division test."""

    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileDivsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_divs(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [64, 64])
                tile_c = pl.op.block.divs(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_divs(a)
                return out_c

        return TileDivsProgram


class TestTileDivs128x128(TestTileDivs):
    """128x128 tile-scalar division test."""

    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileDivsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_divs(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, [0, 0], [128, 128])
                tile_c = pl.op.block.divs(tile_a, 2.0)
                out_c = pl.op.block.store(tile_c, [0, 0], [128, 128], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_divs(a)
                return out_c

        return TileDivsProgram


class TestTileAddWithPTOAS(TestTileAdd128x128):
    """Test tile add with PTOAS optimization strategy.

    This demonstrates how to use a custom optimization strategy.
    """

    def get_strategy(self):
        from pypto.ir.pass_manager import OptimizationStrategy

        return OptimizationStrategy.PTOAS

    def get_name(self) -> str:
        return f"tile_add_ptoas_{self.ROWS}x{self.COLS}"


class TestCustomArrayInit(PTOTestCase):
    """Test case demonstrating custom array initialization patterns."""

    def get_name(self) -> str:
        return "custom_array_init"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            # Small array: custom values (will be serialized)
            TensorSpec(
                "small",
                [3, 3],
                DataType.FP32,
                init_value=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            ),
            # Identity matrix
            TensorSpec("identity", [4, 4], DataType.FP32, init_value=np.eye(4, dtype=np.float32)),
            # Constant array (optimized to np.full)
            TensorSpec("constant", [5, 5], DataType.FP32, init_value=np.ones((5, 5)) * 3.14),
            # Diagonal matrix (small arrays will be serialized)
            TensorSpec(
                "diagonal", [3, 3], DataType.FP32, init_value=np.diag([1, 2, 3]).astype(np.float32)
            ),
            # Output
            TensorSpec("out", [3, 3], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Placeholder - this test is just for demonstrating array initialization
        return None

    def compute_expected(self, tensors, params=None):
        # Simple example: copy small array to output
        tensors["out"][:] = tensors["small"][:3, :3]


# =============================================================================
# pytest test functions
# =============================================================================


class TestElementwiseOperations:
    """Test suite for elementwise operations."""

    # Binary tile-tile operations
    def test_tile_add_64x64(self, test_runner):
        """Test tile addition with 64x64 shape."""
        test_case = TestTileAdd64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_add_128x128(self, test_runner):
        """Test tile addition with 128x128 shape."""
        test_case = TestTileAdd128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_sub_64x64(self, test_runner):
        """Test tile subtraction with 64x64 shape."""
        test_case = TestTileSub64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_sub_128x128(self, test_runner):
        """Test tile subtraction with 128x128 shape."""
        test_case = TestTileSub128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_mul_64x64(self, test_runner):
        """Test tile multiplication with 64x64 shape."""
        test_case = TestTileMul64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_mul_128x128(self, test_runner):
        """Test tile multiplication with 128x128 shape."""
        test_case = TestTileMul128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_div_64x64(self, test_runner):
        """Test tile division with 64x64 shape."""
        test_case = TestTileDiv64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_div_128x128(self, test_runner):
        """Test tile division with 128x128 shape."""
        test_case = TestTileDiv128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_maximum_64x64(self, test_runner):
        """Test tile element-wise maximum with 64x64 shape."""
        test_case = TestTileMaximum64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_maximum_128x128(self, test_runner):
        """Test tile element-wise maximum with 128x128 shape."""
        test_case = TestTileMaximum128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Scalar operations
    def test_tile_adds_64x64(self, test_runner):
        """Test tile-scalar addition with 64x64 shape."""
        test_case = TestTileAdds64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_adds_128x128(self, test_runner):
        """Test tile-scalar addition with 128x128 shape."""
        test_case = TestTileAdds128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_subs_64x64(self, test_runner):
        """Test tile-scalar subtraction with 64x64 shape."""
        test_case = TestTileSubs64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_subs_128x128(self, test_runner):
        """Test tile-scalar subtraction with 128x128 shape."""
        test_case = TestTileSubs128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_muls_64x64(self, test_runner):
        """Test tile-scalar multiplication with 64x64 shape."""
        test_case = TestTileMuls64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_muls_128x128(self, test_runner):
        """Test tile-scalar multiplication with 128x128 shape."""
        test_case = TestTileMuls128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_divs_64x64(self, test_runner):
        """Test tile-scalar division with 64x64 shape."""
        test_case = TestTileDivs64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 64x64: {result.error}"

    def test_tile_divs_128x128(self, test_runner):
        """Test tile-scalar division with 128x128 shape."""
        test_case = TestTileDivs128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    # Optimization strategy test
    def test_tile_add_ptoas_strategy(self, test_runner):
        """Test tile addition with PTOAS optimization strategy."""
        test_case = TestTileAddWithPTOAS()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"
