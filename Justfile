set shell := ["bash", "-cu"]

default:
    @just --list

install:
    uv sync --group dev --frozen
		uv run --frozen pre-commit install

docs-serve: install
    uv run --frozen sphinx-autobuild docs docs/_build/html --open-browser

docs-build: install docs-clean
    uv run --frozen sphinx-build -b html docs docs/_build/html

docs-clean:
    rm -rf docs/_build docs/jupyter_execute

test:
    uv run --frozen pytest --cov=./ --cov-report=xml
