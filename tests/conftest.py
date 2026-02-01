"""
pytest configuration and fixtures for PTO testing framework.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
_SRC_PATH = Path(__file__).parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from pto_test.core.test_case import TestConfig
from pto_test.core.test_runner import TestRunner


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--platform",
        action="store",
        default="a2a3sim",
        choices=["a2a3sim", "a2a3"],
        help="Target platform for tests (default: a2a3sim)",
    )
    parser.addoption(
        "--device",
        action="store",
        default=0,
        type=int,
        help="Device ID for hardware tests (default: 0)",
    )
    parser.addoption(
        "--fuzz-count",
        action="store",
        default=10,
        type=int,
        help="Number of fuzz test iterations (default: 10)",
    )
    parser.addoption(
        "--fuzz-seed",
        action="store",
        default=None,
        type=int,
        help="Random seed for fuzz tests (default: random)",
    )


@pytest.fixture
def test_config(request) -> TestConfig:
    """Fixture providing test configuration from command line options."""
    return TestConfig(
        platform=request.config.getoption("--platform"),
        device_id=request.config.getoption("--device"),
    )


@pytest.fixture
def test_runner(test_config) -> TestRunner:
    """Fixture providing a test runner instance."""
    return TestRunner(test_config)


@pytest.fixture
def fuzz_count(request) -> int:
    """Fixture providing fuzz test iteration count."""
    return request.config.getoption("--fuzz-count")


@pytest.fixture
def fuzz_seed(request) -> int:
    """Fixture providing fuzz test seed."""
    seed = request.config.getoption("--fuzz-seed")
    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)
    return seed


# Standard test shapes for parameterized tests
STANDARD_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
]


@pytest.fixture(params=STANDARD_SHAPES)
def tensor_shape(request):
    """Parameterized fixture for tensor shapes."""
    return list(request.param)


# Skip markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "fuzz: mark test as fuzz test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on platform."""
    platform = config.getoption("--platform")

    skip_hardware = pytest.mark.skip(reason="hardware tests require --platform=a2a3")

    for item in items:
        if "hardware" in item.keywords and platform != "a2a3":
            item.add_marker(skip_hardware)
