"""
PTO Testing Framework

End-to-end testing framework for PyPTO frontend and Simpler runtime.
"""

# Codegen module exports
from pto_test.codegen import (
    ConfigGenerator,
    GoldenGenerator,
    OrchGenerator,
    ProgramCodeGenerator,
)
from pto_test.core.test_case import (
    DataType,
    PTOTestCase,
    TensorSpec,
    TestConfig,
    TestResult,
)
from pto_test.core.test_runner import TestRunner, TestSuite
from pto_test.core.validators import ResultValidator

__version__ = "0.1.0"
__all__ = [
    # Core
    "PTOTestCase",
    "TensorSpec",
    "TestConfig",
    "TestResult",
    "DataType",
    "TestRunner",
    "TestSuite",
    "ResultValidator",
    # Codegen
    "ProgramCodeGenerator",
    "OrchGenerator",
    "ConfigGenerator",
    "GoldenGenerator",
]
