"""
Tests for elementwise operations using PyPTO frontend.

Tests tile-level binary operations like add, sub, mul, div.
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


class TestTileAdd(PTOTestCase):
    """Test case for tile element-wise addition."""

    def __init__(self, rows: int = 128, cols: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"tile_add_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        rows, cols = self.rows, self.cols

        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                c: pl.Tensor[[rows, cols], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, rows, cols)
                tile_b = pl.op.block.load(b, 0, 0, rows, cols)
                tile_c = pl.op.block.add(tile_a, tile_b)
                pl.op.block.store(tile_c, 0, 0, rows, cols, c)

        return TileAddProgram

    def get_orchestration(self) -> str:
        return f'''
#include "runtime.h"
#include <iostream>

extern "C" {{

int build_test_graph(Runtime* runtime, uint64_t* args, int arg_count) {{
    // Extract arguments: [ptr_a, ptr_b, ptr_c, size_a, size_b, size_c, total_size]
    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    void* host_c = reinterpret_cast<void*>(args[2]);
    size_t size_a = static_cast<size_t>(args[3]);
    size_t size_b = static_cast<size_t>(args[4]);
    size_t size_c = static_cast<size_t>(args[5]);
    int SIZE = static_cast<int>(args[6]);

    std::cout << "\\n=== build_test_graph: Creating Task Runtime ===\\n";

    // Allocate device memory
    void* dev_a = runtime->host_api.device_malloc(size_a);
    void* dev_b = runtime->host_api.device_malloc(size_b);
    void* dev_c = runtime->host_api.device_malloc(size_c);

    // Copy inputs to device
    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    runtime->host_api.copy_to_device(dev_b, host_b, size_b);

    // Register output for copy-back
    runtime->record_tensor_pair(host_c, dev_c, size_c);

    // Build task args: [ptr_a, ptr_b, ptr_c, SIZE]
    uint64_t task_args[4];
    task_args[0] = reinterpret_cast<uint64_t>(dev_a);
    task_args[1] = reinterpret_cast<uint64_t>(dev_b);
    task_args[2] = reinterpret_cast<uint64_t>(dev_c);
    task_args[3] = SIZE;

    // Add task (func_id=0, core_type=1 for AIV)
    int t0 = runtime->add_task(task_args, 4, 0, 1);
    std::cout << "Added task " << t0 << ": {self.get_name()}\\n";

    std::cout << "Created runtime with " << runtime->get_task_count() << " tasks\\n";
    return 0;
}}

}}  // extern "C"
'''

    def compute_expected(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {"c": inputs["a"] + inputs["b"]}


class TestTileMul(PTOTestCase):
    """Test case for tile element-wise multiplication."""

    def __init__(self, rows: int = 128, cols: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"tile_mul_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        rows, cols = self.rows, self.cols

        @pl.program
        class TileMulProgram:
            @pl.function
            def tile_mul(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                c: pl.Tensor[[rows, cols], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, rows, cols)
                tile_b = pl.op.block.load(b, 0, 0, rows, cols)
                tile_c = pl.op.block.mul(tile_a, tile_b)
                pl.op.block.store(tile_c, 0, 0, rows, cols, c)

        return TileMulProgram

    def get_orchestration(self) -> str:
        return f'''
#include "runtime.h"
#include <iostream>

extern "C" {{

int build_test_graph(Runtime* runtime, uint64_t* args, int arg_count) {{
    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    void* host_c = reinterpret_cast<void*>(args[2]);
    size_t size_a = static_cast<size_t>(args[3]);
    size_t size_b = static_cast<size_t>(args[4]);
    size_t size_c = static_cast<size_t>(args[5]);
    int SIZE = static_cast<int>(args[6]);

    void* dev_a = runtime->host_api.device_malloc(size_a);
    void* dev_b = runtime->host_api.device_malloc(size_b);
    void* dev_c = runtime->host_api.device_malloc(size_c);

    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    runtime->host_api.copy_to_device(dev_b, host_b, size_b);
    runtime->record_tensor_pair(host_c, dev_c, size_c);

    uint64_t task_args[4];
    task_args[0] = reinterpret_cast<uint64_t>(dev_a);
    task_args[1] = reinterpret_cast<uint64_t>(dev_b);
    task_args[2] = reinterpret_cast<uint64_t>(dev_c);
    task_args[3] = SIZE;

    runtime->add_task(task_args, 4, 0, 1);
    return 0;
}}

}}
'''

    def compute_expected(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {"c": inputs["a"] * inputs["b"]}


class TestElementwiseOperations:
    """Test suite for elementwise operations."""

    def test_tile_add_basic(self, test_runner):
        """Test basic tile addition."""
        test_case = TestTileAdd(rows=128, cols=128)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_mul_basic(self, test_runner):
        """Test basic tile multiplication."""
        test_case = TestTileMul(rows=128, cols=128)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128)])
    def test_tile_add_shapes(self, test_runner, rows, cols):
        """Test tile addition with various shapes."""
        test_case = TestTileAdd(rows=rows, cols=cols)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for {rows}x{cols}: {result.error}"
