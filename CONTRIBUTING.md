
# CONTRIBUTING

Setup of dev environment:
```
# install external dependencies
uv sync --extra dev

# install git pre-commit hook
uv run pre-commit install

# run linter
uv run ruff check --fix

# run formatter
uv run ruff format

# run static-type checker
uv run mypy

# run unit tests
uv run mypy

# generate coverage report
# will be located in htmlcov/
uv run mypy --cov=word_frequency --cov-report=html
```

Periodically we will need to update our dependencies:

```
# update dependencies and the lockfile
uv sync -U

# update the pinned versions in the git hook
uv run pre-commit autoupdate
```
