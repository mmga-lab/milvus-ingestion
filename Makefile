.PHONY: help install install-dev format lint test test-cov clean build publish

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pdm install --prod

install-dev:  ## Install development dependencies
	pdm install

format:  ## Format code with ruff
	pdm run ruff format src tests

lint:  ## Run linting checks
	pdm run ruff check src tests
	pdm run mypy src

test:  ## Run tests
	pdm run pytest

test-cov:  ## Run tests with coverage
	pdm run pytest --cov=src --cov-report=term-missing --cov-report=html

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	pdm build

publish:  ## Publish to PyPI (requires PDM_PUBLISH_TOKEN)
	pdm publish

check: lint test  ## Run all checks (lint + test)