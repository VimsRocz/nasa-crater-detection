# Contributing

Thanks for your interest in improving this project!

## Development Setup

Prerequisites:
- Python 3.12+

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Linting / Formatting

```bash
python -m ruff check .
python -m ruff format .
```

## Tests

```bash
python -m pytest
```

## Pull Requests

- Keep PRs focused and small when possible.
- Update `README.md` when behavior or usage changes.
- If you add functionality, add tests in `tests/`.
