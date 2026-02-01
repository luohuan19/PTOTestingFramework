# PTO Testing Framework

A general-purpose integration testing framework designed to perform fuzz testing between different frontends and backends.

## Getting Started

This repository uses git submodules. Clone with the `--recursive` flag to fetch all dependencies:

```bash
git clone --recursive https://github.com/<your-org>/pto-testing-framework.git
```

If you have already cloned the repository without `--recursive`, run:

```bash
git submodule update --init --recursive
```

## Architecture

The framework decouples frontends and backends, allowing any frontend to be tested against any backend via fuzz testing.

### Frontend

- [pypto](https://github.com/hw-native-sys/pypto) — located at `3rdparty/pypto`

### Backend

- [simpler](https://github.com/ChaoWao/simpler) — located at `3rdparty/simpler`

## Directory Structure

```
pto-testing-framework/
├── src/pto_test/
│   ├── core/               # Core test infrastructure
│   │   ├── test_case.py    # Test case base classes
│   │   ├── test_runner.py  # Test execution engine
│   │   └── validators.py   # Result validation
│   ├── codegen/            # Code generation
│   │   ├── kernel_generator.py
│   │   └── orch_generator.py
│   └── fuzzing/            # Fuzz testing
│       └── generators.py
├── tests/                  # Test cases
│   └── test_cases/
├── 3rdparty/               # Dependencies
│   ├── pypto/              # PyPTO submodule
│   └── simpler/            # Simpler submodule
└── pyproject.toml
```

## Installation

```bash
pip install -e .
```

## Running Tests

### Simulation Platform (Default)

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cases/test_elementwise.py -v

# Run fuzz tests with custom count
pytest tests/test_cases/test_fuzz.py -v --fuzz-count=100

# Run fuzz tests with fixed seed (for reproducibility)
pytest tests/test_cases/test_fuzz.py -v --fuzz-seed=42
```

### Hardware Platform

```bash
# Run on real hardware (requires NPU device)
pytest tests/ -v --platform=a2a3 --device=0
```

## Writing Test Cases

```python
from pto_test.core.test_case import PTOTestCase, TensorSpec, DataType

class TestMyOp(PTOTestCase):
    def get_name(self):
        return "my_op_test"

    def define_tensors(self):
        return [
            TensorSpec("input", [128, 128], DataType.FP32, init_value=1.0),
            TensorSpec("output", [128, 128], DataType.FP32, is_output=True),
        ]

    def build_ir(self, ib):
        # Build PyPTO IR (optional for simulation)
        return None

    def compute_expected(self, inputs):
        # Compute expected result with NumPy
        return {"output": inputs["input"] * 2}

    def generate_sim_loop(self):
        # Generate C++ loop for simulation
        return '''for (int i = 0; i < size; i++) {
    output[i] = input[i] * 2;
}'''
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--platform` | a2a3sim | Target platform (a2a3sim or a2a3) |
| `--device` | 0 | Device ID for hardware tests |
| `--fuzz-count` | 10 | Number of fuzz test iterations |
| `--fuzz-seed` | random | Random seed for reproducibility |

## Supported Operations

### Elementwise Binary Operations
- `add`, `sub`, `mul`, `div`

### Scalar Operations
- `adds`, `subs`, `muls`, `divs`

### Unary Operations
- `exp`, `sqrt`, `neg`, `abs`

## Test Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Case Definition                    │
│    (TensorSpec + build_ir + compute_expected)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌───────────────────┐ ┌─────────────────┐
│ 1. Build IR     │ │ 2. Generate Kernel│ │ 3. Prepare Data │
│ (PyPTO)         │ │ (KernelGenerator) │ │ (NumPy arrays)  │
└────────┬────────┘ └─────────┬─────────┘ └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ 4. Compile Components  │
                 │ - Kernel → .o/.text    │
                 │ - Orchestration → .so  │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ 5. Execute on Simpler  │
                 │ - bind_host_binary()   │
                 │ - register_kernel()    │
                 │ - launch_runtime()     │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ 6. Validate Results    │
                 │ - Compare with NumPy   │
                 │ - Return TestResult    │
                 └────────────────────────┘
```

## License

Apache-2.0
