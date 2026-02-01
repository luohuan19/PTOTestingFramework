"""
PTO Testing Framework

End-to-end testing framework for PyPTO frontend and Simpler runtime.
"""

from pto_test.core.test_case import PTOTestCase, TensorSpec, TestConfig, TestResult
from pto_test.core.test_runner import TestRunner
from pto_test.core.validators import ResultValidator

__version__ = "0.1.0"
__all__ = [
    "PTOTestCase",
    "TensorSpec",
    "TestConfig",
    "TestResult",
    "TestRunner",
    "ResultValidator",
]
