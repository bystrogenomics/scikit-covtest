# Testing scikit-covtest

This document provides instructions on how to run tests for `scikit-covtest`.

## Prerequisites

Ensure you have the development dependencies installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
pip install pytest
```

## Running Tests

We use `pytest` for testing. To run all tests, simply execute:

```bash
pytest
```

### Running Specific Tests

To run tests for a specific module, pass the file path to `pytest`:

```bash
pytest tests/test_methods_identity.py
```

To run a specific test case:

```bash
pytest tests/test_methods_identity.py::test_fisher_identity_output
```

## Test Structure

The tests are located in the `tests/` directory:

- `test_methods_identity.py`: Tests for one-sample identity tests.
- `test_methods_spherical.py`: Tests for one-sample sphericity tests.
- `test_methods_two_sample.py`: Tests for two-sample equality tests.
- `test_methods_proportionality.py`: Tests for multi-sample proportionality tests.
- `test_generate_data.py`: Tests for data generation utilities.

## Continuous Integration

Tests are automatically run on GitHub Actions for every push and pull request.
