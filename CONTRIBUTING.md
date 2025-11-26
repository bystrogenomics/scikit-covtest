# Contributing to scikit-covtest

Thank you for your interest in contributing to `scikit-covtest`! We welcome contributions from the community.

## Getting Started

1. **Fork the repository**: Click the "Fork" button on the GitHub page.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/scikit-covtest.git
   cd scikit-covtest
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

## Development Workflow

1. **Create a branch**: Always work on a new branch for your changes.
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make changes**: Write your code and tests.
3. **Run tests**: Ensure all tests pass.
   ```bash
   pytest
   ```
4. **Commit changes**: Write clear and concise commit messages.
   ```bash
   git commit -m "Add new feature X"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```
6. **Submit a Pull Request**: Go to the original repository and open a Pull Request.

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
- Use descriptive variable and function names.
- Add docstrings to all public functions and classes.

## Reporting Bugs

If you find a bug, please open an issue on GitHub with the following details:
- Description of the bug.
- Steps to reproduce.
- Expected vs. actual behavior.
- Environment details (OS, Python version, package version).

## Requesting Features

We welcome feature requests! Please open an issue describing the feature and its use case.
