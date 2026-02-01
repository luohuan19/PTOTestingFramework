"""
Test runner for executing PTO test cases.

Orchestrates the full test execution pipeline:
1. Get program from test case (@pl.program or IRBuilder)
2. Generate kernel code via PyPTO (PassManager → CceCodegen)
3. Compile kernel and orchestration
4. Execute on Simpler runtime
5. Validate results
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pto_test.core.test_case import PTOTestCase, TestConfig, TestResult, TensorSpec
from pto_test.core.validators import ResultValidator

# Add pypto and simpler to path
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
_SIMPLER_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "simpler"
_SIMPLER_PYTHON = _SIMPLER_ROOT / "python"

if _PYPTO_ROOT.exists() and str(_PYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYPTO_ROOT))

if _SIMPLER_PYTHON.exists() and str(_SIMPLER_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SIMPLER_PYTHON))


class TestRunner:
    """Executes PTO test cases on Simpler runtime.

    Handles the complete test execution pipeline including:
    - Kernel code generation via PyPTO
    - Compilation
    - Runtime execution
    - Result validation
    """

    def __init__(self, config: Optional[TestConfig] = None):
        """Initialize test runner.

        Args:
            config: Test configuration. If None, uses default config.
        """
        self.config = config or TestConfig()
        self._runtime_builder = None
        self._pto_compiler = None
        self._host_binary = None
        self._aicpu_binary = None
        self._aicore_binary = None
        self._initialized = False
        self._pass_manager = None
        self._codegen = None

    def _lazy_init(self):
        """Lazy initialization of runtime builder and compiler."""
        if self._initialized:
            return

        try:
            from runtime_builder import RuntimeBuilder
        except ImportError as e:
            raise ImportError(
                f"Cannot import Simpler runtime builder: {e}\n"
                f"Make sure simpler is properly installed at {_SIMPLER_ROOT}"
            )

        self._runtime_builder = RuntimeBuilder(platform=self.config.platform)
        self._pto_compiler = self._runtime_builder.get_pto_compiler()

        # Build runtime binaries once
        print(f"Building runtime for platform: {self.config.platform}")
        self._host_binary, self._aicpu_binary, self._aicore_binary = (
            self._runtime_builder.build("host_build_graph")
        )

        self._initialized = True

    def _get_pass_manager(self):
        """Get or create PassManager with XPlatform strategy."""
        if self._pass_manager is None:
            from pypto.ir.pass_manager import PassManager, OptimizationStrategy
            self._pass_manager = PassManager.get_strategy(OptimizationStrategy.XPlatform)
        return self._pass_manager

    def _get_codegen(self):
        """Get or create CceCodegen instance."""
        if self._codegen is None:
            from pypto.pypto_core import codegen
            self._codegen = codegen.CceCodegen()
        return self._codegen

    def run(self, test_case: PTOTestCase) -> TestResult:
        """Run a test case and return results.

        Args:
            test_case: The test case to run.

        Returns:
            TestResult with pass/fail status and details.
        """
        start_time = time.time()
        test_name = test_case.get_name()

        try:
            self._lazy_init()

            # 1. Get program from test case
            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            # 2. Generate kernel code via PyPTO pipeline
            kernel_code = self._generate_kernel(program)

            # 3. Compile kernel
            kernel_binary = self._compile_kernel(kernel_code, test_name)

            # 4. Get and compile orchestration
            orch_code = test_case.get_orchestration()
            if orch_code is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_orchestration() "
                    "to return orchestration C++ code"
                )
            orch_binary = self._compile_orchestration(orch_code)

            # 5. Prepare test data
            inputs = test_case.prepare_inputs()
            outputs = test_case.prepare_outputs()

            # 6. Execute on runtime
            actual_outputs = self._execute(
                kernel_binary,
                orch_binary,
                inputs,
                outputs,
                test_case,
            )

            # 7. Validate results
            expected = test_case.compute_expected(inputs)
            validator = ResultValidator(
                atol=self.config.atol,
                rtol=self.config.rtol,
            )

            result = validator.validate(expected, actual_outputs, test_name)
            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            return TestResult(
                passed=False,
                test_name=test_name,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _generate_kernel(self, program: Any) -> str:
        """Generate kernel code using PyPTO pipeline.

        Pipeline: Program → PassManager (XPlatform) → CceCodegen → C++

        Args:
            program: PyPTO Program (from @pl.program or ir.Program).

        Returns:
            Generated C++ kernel code string.
        """
        # Apply passes
        pm = self._get_pass_manager()
        transformed = pm.run_passes(program)

        # Generate code for each function
        cg = self._get_codegen()
        code_parts = []
        for func_name, func in transformed.functions.items():
            code = cg.Generate(func)
            code_parts.append(code)

        return "\n\n".join(code_parts)

    def _compile_kernel(self, code: str, kernel_name: str) -> bytes:
        """Compile kernel code to binary.

        Args:
            code: C++ kernel source code.
            kernel_name: Name of the kernel function.

        Returns:
            Binary data (extracted .text section).
        """
        from elf_parser import extract_text_section

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.cpp', delete=False
        ) as f:
            f.write(code)
            source_path = f.name

        try:
            # Compile using PTO compiler
            kernel_o = self._pto_compiler.compile_incore(
                source_path,
                core_type="aiv",
                pto_isa_root=str(_FRAMEWORK_ROOT / "3rdparty" / "pto-isa"),
            )

            # Extract .text section
            return extract_text_section(kernel_o)
        finally:
            Path(source_path).unlink(missing_ok=True)

    def _compile_orchestration(self, code: str) -> bytes:
        """Compile orchestration code to shared library.

        Args:
            code: C++ orchestration source code.

        Returns:
            Shared library binary data.
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.cpp', delete=False
        ) as f:
            f.write(code)
            source_path = f.name

        try:
            # Get include directories
            runtime_include = (
                _SIMPLER_ROOT / "src" / "runtime" / "host_build_graph" / "runtime"
            )
            include_dirs = [str(runtime_include)]
            include_dirs.extend(self._pto_compiler.get_platform_include_dirs())

            # Compile
            return self._pto_compiler.compile_orchestration(
                source_path,
                extra_include_dirs=include_dirs,
            )
        finally:
            Path(source_path).unlink(missing_ok=True)

    def _execute(
        self,
        kernel_binary: bytes,
        orch_binary: bytes,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        test_case: PTOTestCase,
    ) -> Dict[str, np.ndarray]:
        """Execute test on Simpler runtime.

        Args:
            kernel_binary: Compiled kernel binary.
            orch_binary: Compiled orchestration binary.
            inputs: Input arrays.
            outputs: Output arrays (will be modified in place).
            test_case: Test case for tensor specifications.

        Returns:
            Dict of output arrays with computed values.
        """
        from bindings import (
            bind_host_binary,
            register_kernel,
            set_device,
            launch_runtime,
        )

        # Load runtime
        Runtime = bind_host_binary(self._host_binary)

        # Set device
        set_device(self.config.device_id)

        # Register kernel
        register_kernel(0, kernel_binary)

        # Build func_args
        # Format: [ptr1, ptr2, ..., size1, size2, ..., total_size]
        func_args = []
        tensor_specs = test_case.tensor_specs

        # Add pointers
        for spec in tensor_specs:
            if spec.is_output:
                arr = outputs[spec.name]
            else:
                arr = inputs[spec.name]
            func_args.append(arr.ctypes.data)

        # Add sizes
        for spec in tensor_specs:
            if spec.is_output:
                arr = outputs[spec.name]
            else:
                arr = inputs[spec.name]
            func_args.append(arr.nbytes)

        # Add total element count (use first tensor's size)
        func_args.append(tensor_specs[0].size)

        # Create and initialize runtime
        runtime = Runtime()
        runtime.initialize(orch_binary, "build_test_graph", func_args)

        # Execute
        launch_runtime(
            runtime,
            aicpu_thread_num=self.config.aicpu_thread_num,
            block_dim=self.config.block_dim,
            device_id=self.config.device_id,
            aicpu_binary=self._aicpu_binary,
            aicore_binary=self._aicore_binary,
        )

        # Finalize (copies results back)
        runtime.finalize()

        return outputs


class TestSuite:
    """Collection of test cases that can be run together."""

    def __init__(self, name: str, config: Optional[TestConfig] = None):
        """Initialize test suite.

        Args:
            name: Suite name.
            config: Configuration for all tests in suite.
        """
        self.name = name
        self.config = config or TestConfig()
        self._test_cases: list = []

    def add_test(self, test_case: PTOTestCase) -> "TestSuite":
        """Add a test case to the suite.

        Args:
            test_case: Test case to add.

        Returns:
            Self for chaining.
        """
        self._test_cases.append(test_case)
        return self

    def run_all(self, runner: Optional[TestRunner] = None) -> Dict[str, TestResult]:
        """Run all test cases in the suite.

        Args:
            runner: Test runner to use. If None, creates one.

        Returns:
            Dict mapping test names to results.
        """
        if runner is None:
            runner = TestRunner(self.config)

        results = {}
        for test_case in self._test_cases:
            result = runner.run(test_case)
            results[test_case.get_name()] = result
            print(result)

        return results

    def summary(self, results: Dict[str, TestResult]) -> str:
        """Generate summary of test results.

        Args:
            results: Results from run_all().

        Returns:
            Summary string.
        """
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        failed = total - passed

        lines = [
            f"\n{'=' * 50}",
            f"Test Suite: {self.name}",
            f"{'=' * 50}",
            f"Passed: {passed}/{total}",
            f"Failed: {failed}/{total}",
        ]

        if failed > 0:
            lines.append("\nFailed tests:")
            for name, result in results.items():
                if not result.passed:
                    lines.append(f"  - {name}: {result.error}")

        return "\n".join(lines)
