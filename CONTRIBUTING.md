# Contributing to proxymate

We want to make contributing to this project as easy and transparent as
possible. We welcome your contributions!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/facebookresearch/proxymate.git
cd proxymate

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
```

## Code Style

We use [Black](https://github.com/psf/black) for formatting and [isort](https://pycqa.github.io/isort/) for import sorting:

```bash
black .
isort .
flake8 --max-line-length 88 --extend-ignore E203,W503 .
```

## Adding a New Check

All checks inherit from `CheckBase` in `checks/base.py`. To add a new check:

1. Choose the appropriate level module (e.g., `checks/unit_level/precision.py`).
2. Subclass `CheckBase` and implement `compute()`, `project()`, and `get_default_threshold()`.
3. Add tests in the corresponding `tests/` directory.
4. Document the check in `website/docs/api/checks.md`.

## Building Documentation

The documentation site is built with [Docusaurus](https://docusaurus.io/) and
lives under `website/`:

```bash
cd website
npm install
npm start
```

The documentation will be available at `http://localhost:3000`.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes and code is formatted.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use [GitHub Issues](https://github.com/facebookresearch/proxymate/issues) to
track public bugs and feature requests. Please ensure your description is clear
and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License

By contributing to proxymate, you agree that your contributions will be licensed
under the BSD-3-Clause License found in the LICENSE file in the root directory of
this source tree.
