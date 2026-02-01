"""
Tests for scalar operations using PyPTO frontend.

Tests tile-scalar operations like adds, subs, muls, divs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# Add pypto to path
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
if _PYPTO_ROOT.exists() and str(_PYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYPTO_ROOT))


class TestTileAddScalar(PTOTestCase):
    """Test case for tile + scalar addition."""

    def __init__(self, rows: int = 128, cols: int = 128, scalar: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.scalar = scalar

    def get_name(self) -> str:
        return f"tile_adds_{self.rows}x{self.cols}_s{self.scalar}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=5.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        rows, cols = self.rows, self.cols
        scalar = self.scalar

        @pl.program
        class TileAddsProgram:
            @pl.function
            def tile_adds(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, rows, cols)
                tile_b = pl.op.block.adds(tile_a, scalar)
                pl.op.block.store(tile_b, 0, 0, rows, cols, b)

        return TileAddsProgram

    def get_orchestration(self) -> str:
        return '''
#include "runtime.h"
#include <iostream>

extern "C" {

int build_test_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    size_t size_a = static_cast<size_t>(args[2]);
    size_t size_b = static_cast<size_t>(args[3]);
    int SIZE = static_cast<int>(args[4]);

    void* dev_a = runtime->host_api.device_malloc(size_a);
    void* dev_b = runtime->host_api.device_malloc(size_b);

    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    runtime->record_tensor_pair(host_b, dev_b, size_b);

    uint64_t task_args[3];
    task_args[0] = reinterpret_cast<uint64_t>(dev_a);
    task_args[1] = reinterpret_cast<uint64_t>(dev_b);
    task_args[2] = SIZE;

    runtime->add_task(task_args, 3, 0, 1);
    return 0;
}

}
'''

    def compute_expected(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {"b": inputs["a"] + self.scalar}


class TestTileMulScalar(PTOTestCase):
    """Test case for tile * scalar multiplication."""

    def __init__(self, rows: int = 128, cols: int = 128, scalar: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.scalar = scalar

    def get_name(self) -> str:
        return f"tile_muls_{self.rows}x{self.cols}_s{self.scalar}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        rows, cols = self.rows, self.cols
        scalar = self.scalar

        @pl.program
        class TileMulsProgram:
            @pl.function
            def tile_muls(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, rows, cols)
                tile_b = pl.op.block.muls(tile_a, scalar)
                pl.op.block.store(tile_b, 0, 0, rows, cols, b)

        return TileMulsProgram

    def get_orchestration(self) -> str:
        return '''
#include "runtime.h"

extern "C" {

int build_test_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    size_t size_a = static_cast<size_t>(args[2]);
    size_t size_b = static_cast<size_t>(args[3]);
    int SIZE = static_cast<int>(args[4]);

    void* dev_a = runtime->host_api.device_malloc(size_a);
    void* dev_b = runtime->host_api.device_malloc(size_b);

    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    runtime->record_tensor_pair(host_b, dev_b, size_b);

    uint64_t task_args[3];
    task_args[0] = reinterpret_cast<uint64_t>(dev_a);
    task_args[1] = reinterpret_cast<uint64_t>(dev_b);
    task_args[2] = SIZE;

    runtime->add_task(task_args, 3, 0, 1);
    return 0;
}

}
'''

    def compute_expected(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {"b": inputs["a"] * self.scalar}


class TestScalarOperations:
    """Test suite for scalar operations."""

    def test_tile_add_scalar_basic(self, test_runner):
        """Test basic tile + scalar addition."""
        test_case = TestTileAddScalar(rows=128, cols=128, scalar=1.0)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_mul_scalar_basic(self, test_runner):
        """Test basic tile * scalar multiplication."""
        test_case = TestTileMulScalar(rows=128, cols=128, scalar=2.0)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("scalar", [0.5, 1.0, 2.0, 10.0, -1.0])
    def test_tile_add_scalar_values(self, test_runner, scalar):
        """Test tile + scalar with various scalar values."""
        test_case = TestTileAddScalar(rows=128, cols=128, scalar=scalar)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for scalar={scalar}: {result.error}"
