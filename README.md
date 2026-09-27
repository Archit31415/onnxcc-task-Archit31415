# onnxcc

A readable C++20 ONNX model compiler and CPU inference engine.

## Setup and Build

```bash
# Build project

cmake -S . -B build
cmake --build build --target onnxcc unit_tests

# Run unit tests
ctest --test-dir build --output-on-failure

# Generate test fixtures
python3 scripts/generate_test_models.py
```

## CLI Usage

```bash
# View help
./build/onnxcc --help

# Run dump subcommand
./build/onnxcc dump --model tests/fixtures/mlp.onnx --show-graph --verbose
```
