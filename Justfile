set shell := ["bash", "-cu"]

venv := ".venv-docs"
python := venv + "/bin/python"

default:
    @just --list

docs-install:
    python3 -m venv {{venv}}
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install -e .
    {{python}} -m pip install -r docs/requirements.txt

docs-serve: docs-install
    {{venv}}/bin/sphinx-autobuild docs docs/_build/html --open-browser

docs-build: docs-install
    {{venv}}/bin/sphinx-build -b html docs docs/_build/html

docs-clean:
    rm -rf docs/_build docs/jupyter_execute
