install:
uv pip install -e ".[dev]"

test:
python -m pytest --tb=short

coverage:
python -m pytest --cov=src --cov-report=term-missing

lint:
ruff check src/ tests/

pylint:
pylint src/ --fail-under=7.0

all: install lint test

.PHONY: install test coverage lint pylint all
