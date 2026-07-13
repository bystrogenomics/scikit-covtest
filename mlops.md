# MLOps & CI/CD Technical Specification Sheet

This document specifies the concrete configuration files and implementation details for the proposed MLOps and CI/CD tasks in `scikit-covtest`.

---

## 1. Upgraded CI Workflow (`.github/workflows/skcovtest.yml`)

Upgrade the CI workflow to run on a multi-OS matrix (Ubuntu, macOS, Windows), cache pip dependencies, and verify code quality with `ruff` and `black`.

```yaml
name: CI - Matrix Lint and Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    name: ${{ matrix.os }} / Python ${{ matrix.python-version }}
    runs-on: ${{ matrix.os }}

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip' # Caches pip dependencies automatically

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov ruff black
        pip install -e .

    - name: Check formatting with black
      run: black --check covtest tests

    - name: Lint with ruff
      run: ruff check covtest tests

    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=covtest --cov-report=xml --maxfail=1 --disable-warnings -q

    - name: Upload coverage report (Optional)
      uses: codecov/codecov-action@v4
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        fail_ci_if_error: false
```

---

## 2. Automated CD / Package Release Workflow (`.github/workflows/publish.yml`)

Implement a CD workflow that automatically builds wheels and source distributions, publishing them securely to PyPI via OIDC trusted publishing.

```yaml
name: CD - Build and Publish to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  build:
    name: Build distributions
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install build tool
      run: pip install build

    - name: Build sdist and wheel
      run: python -m build

    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

  publish-pypi:
    name: Publish to PyPI
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write # Required for trusted publishing OIDC authentication
    steps:
    - name: Download artifacts
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

---

## 3. Local Development Reproducibility Configuration

### 3.1 Pre-commit Config (`.pre-commit-config.yaml`)

Add this configuration to the root of the project to check and format files on every commit.

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [ --fix ]
```

### 3.2 Docker Devcontainer Config (`.devcontainer/devcontainer.json`)

Configure a Docker container environment for VS Code.

```json
{
  "name": "Python Scientific Devcontainer",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/git:1": {}
  },
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.formatting.provider": "black",
        "editor.formatOnSave": true
      },
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff"
      ]
    }
  },
  "postCreateCommand": "pip install -r requirements.txt && pip install -e ."
}
```

---

## 4. Weekly Statistical Validation Pipeline (`.github/workflows/weekly-validation.yml`)

Unlike software correctness, statistical models require verifying that their Type-I Error Rates and Power Curves meet asymptotic thresholds under simulation.

```yaml
name: Weekly Statistical Validation

on:
  schedule:
    - cron: '0 0 * * 0' # Every Sunday at midnight
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install . pytest scipy

    - name: Run Monte Carlo validation
      run: |
        python scripts/validate_statistical_correctness.py
```

### 4.1 Validation Script (`scripts/validate_statistical_correctness.py`)

A script to run thousands of simulations verifying the $p$-values are distributed uniformly under the null hypothesis (using a Kolmogorov-Smirnov test).

```python
import numpy as np
import scipy.stats as stats
import sys
from covtest.methods.hypothesis_identity import identity_covariance_test

def check_pvalue_uniformity(n_sims=1000, n=50, p=10, alpha=0.01):
    """Under the null hypothesis, p-values should follow a Uniform(0, 1) distribution.
    We test this using a Kolmogorov-Smirnov (KS) test.
    """
    rng = np.random.default_rng(42)
    p_values = []
    
    for _ in range(n_sims):
        # Generate data under H0 (covariance = Identity)
        X = rng.normal(size=(n, p))
        res = identity_covariance_test(X, method="chen_2010")
        p_values.append(res["p_value"])
        
    # Perform Kolmogorov-Smirnov test against uniform distribution
    ks_stat, ks_pval = stats.kstest(p_values, 'uniform')
    
    print(f"KS test statistic: {ks_stat:.4f}, p-value: {ks_pval:.4f}")
    
    # If the KS p-value is extremely small, the distribution is significantly non-uniform
    if ks_pval < alpha:
        print("ERROR: Test p-value distribution significantly deviates from Uniform(0, 1).")
        return False
        
    print("SUCCESS: Test p-value distribution conforms to Uniform(0, 1).")
    return True

if __name__ == "__main__":
    success = check_pvalue_uniformity()
    if not success:
        sys.exit(1)
```
