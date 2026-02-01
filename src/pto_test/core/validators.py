"""
Result validation utilities.

Provides classes and functions for validating test outputs against
expected results with configurable tolerances.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pto_test.core.test_case import TestResult


class ResultValidator:
    """Validates test results against expected outputs.

    Supports configurable absolute and relative tolerances for
    floating-point comparisons.
    """

    def __init__(self, atol: float = 1e-5, rtol: float = 1e-5, max_mismatch_samples: int = 10):
        """Initialize validator.

        Args:
            atol: Absolute tolerance for comparison.
            rtol: Relative tolerance for comparison.
            max_mismatch_samples: Maximum number of mismatch indices to report.
        """
        self.atol = atol
        self.rtol = rtol
        self.max_mismatch_samples = max_mismatch_samples

    def validate(
        self,
        expected: Dict[str, np.ndarray],
        actual: Dict[str, np.ndarray],
        test_name: str = "test",
    ) -> TestResult:
        """Validate actual results against expected.

        Args:
            expected: Dict mapping output names to expected arrays.
            actual: Dict mapping output names to actual arrays.
            test_name: Name of the test for reporting.

        Returns:
            TestResult with validation details.
        """
        # Check all expected outputs are present
        for name in expected:
            if name not in actual:
                return TestResult(
                    passed=False,
                    test_name=test_name,
                    error=f"Missing output tensor: {name}",
                )

        # Validate each output tensor
        total_mismatch_count = 0
        max_abs_error = 0.0
        max_rel_error = 0.0
        all_mismatch_indices: List[Tuple] = []

        for name in expected:
            exp = expected[name]
            act = actual[name]

            # Check shape
            if exp.shape != act.shape:
                return TestResult(
                    passed=False,
                    test_name=test_name,
                    error=f"Shape mismatch for '{name}': expected {exp.shape}, got {act.shape}",
                )

            # Check values
            result = self._compare_arrays(exp, act, name)
            if result is not None:
                return result

            # Collect error statistics
            abs_diff = np.abs(exp - act)
            tensor_max_abs = np.max(abs_diff)
            max_abs_error = max(max_abs_error, tensor_max_abs)

            # Relative error (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = abs_diff / (np.abs(exp) + 1e-10)
                tensor_max_rel = np.nanmax(rel_diff)
            max_rel_error = max(max_rel_error, tensor_max_rel)

            # Count mismatches
            close = np.isclose(exp, act, atol=self.atol, rtol=self.rtol)
            mismatches = np.argwhere(~close)
            total_mismatch_count += len(mismatches)
            all_mismatch_indices.extend(
                [(name, tuple(idx)) for idx in mismatches[: self.max_mismatch_samples]]
            )

        # Determine pass/fail
        passed = total_mismatch_count == 0

        return TestResult(
            passed=passed,
            test_name=test_name,
            error=None if passed else f"{total_mismatch_count} element(s) mismatch",
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            mismatch_count=total_mismatch_count,
            mismatch_indices=all_mismatch_indices[: self.max_mismatch_samples] if not passed else None,
        )

    def _compare_arrays(
        self, expected: np.ndarray, actual: np.ndarray, name: str
    ) -> Optional[TestResult]:
        """Compare two arrays and return error result if they differ significantly.

        Returns None if arrays match within tolerance.
        """
        # Handle special cases
        if expected.size == 0:
            return None  # Empty arrays always match

        # Check for NaN/Inf
        if np.any(np.isnan(actual)) and not np.any(np.isnan(expected)):
            nan_count = np.sum(np.isnan(actual))
            return TestResult(
                passed=False,
                test_name=name,
                error=f"Unexpected NaN values in '{name}': {nan_count} elements",
            )

        if np.any(np.isinf(actual)) and not np.any(np.isinf(expected)):
            inf_count = np.sum(np.isinf(actual))
            return TestResult(
                passed=False,
                test_name=name,
                error=f"Unexpected Inf values in '{name}': {inf_count} elements",
            )

        return None

    def validate_single(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        name: str = "output",
    ) -> TestResult:
        """Validate a single output array.

        Convenience method for validating a single tensor.
        """
        return self.validate({name: expected}, {name: actual}, name)


def assert_close(
    expected: np.ndarray,
    actual: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name: str = "output",
) -> None:
    """Assert that two arrays are close within tolerance.

    Raises AssertionError with detailed message if arrays differ.

    Args:
        expected: Expected array.
        actual: Actual array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        name: Name for error reporting.
    """
    validator = ResultValidator(atol=atol, rtol=rtol)
    result = validator.validate_single(expected, actual, name)

    if not result.passed:
        msg = f"Arrays not close: {result.error}"
        if result.max_abs_error is not None:
            msg += f"\n  Max absolute error: {result.max_abs_error:.6e}"
        if result.max_rel_error is not None:
            msg += f"\n  Max relative error: {result.max_rel_error:.6e}"
        if result.mismatch_indices:
            msg += f"\n  Sample mismatches: {result.mismatch_indices[:5]}"
        raise AssertionError(msg)
