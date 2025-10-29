# Jira Agile Metrics - Makefile
# Provides common development and deployment tasks

.PHONY: help install install-dev clean test lint format check docker-build docker-run webapp run check-full

# Python interpreter
PYTHON = python3
VENV = .venv
VENV_BIN = $(VENV)/bin
PIP = $(VENV_BIN)/pip
PYTEST = $(VENV_BIN)/pytest
PTW = $(VENV_BIN)/ptw
BLACK = $(VENV_BIN)/black
RUFF = $(VENV_BIN)/ruff
PYLINT = $(VENV_BIN)/pylint
MYPY = $(VENV_BIN)/mypy

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Docker image names
DOCKER_IMAGE_DEV = jira-agile-metrics-dev
DOCKER_IMAGE_PROD = jira-agile-metrics
DOCKER_IMAGE_WEBAPP = jira-agile-metrics-webapp

# Container name (configurable via environment variable)
CONTAINER_NAME ?= jira_metrics

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(GREEN)Available make targets:$(NC)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' | grep -E '^(  |dev|install|lint|format|test|run|clean)'
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' | grep 'docker'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install      # Install dependencies"
	@echo "  make test         # Run tests"
	@echo "  make lint         # Run linters"
	@echo "  make format       # Format code with black"
	@echo "  make docker-build # Build Docker images"

## Development targets

venv: ## Create virtual environment
	@if [ ! -d $(VENV) ]; then \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
	fi

install: venv ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt

clean: ## Remove build artifacts and temporary files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete$(NC)"

clean-venv: clean ## Remove virtual environment
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	rm -rf $(VENV)
	@echo "$(GREEN)Virtual environment removed$(NC)"

## Testing targets

test: ## Run tests with pytest
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) -v

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) --cov=jira_agile_metrics --cov-report=html --cov-report=term

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests with verbose output...$(NC)"
	$(PYTEST) -vv

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(PTW)

## Linting and formatting targets

lint: ## Run all linters (ruff and pylint)
	@echo "$(GREEN)Running ruff linter...$(NC)"
	$(RUFF) check .
	@echo "$(GREEN)Running pylint...$(NC)"
	$(PYLINT) jira_agile_metrics/

lint-fix: ## Run ruff with auto-fix
	@echo "$(GREEN)Running ruff with auto-fix...$(NC)"
	$(RUFF) check . --fix

pylint: ## Run pylint
	@echo "$(GREEN)Running pylint...$(NC)"
	$(PYLINT) jira_agile_metrics/

format: ## Format code with black
	@echo "$(GREEN)Formatting code with black...$(NC)"
	$(BLACK) .

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(BLACK) . --check

check: format-check lint ## Run all checks without making changes
	@echo "$(GREEN)All checks passed!$(NC)"

## Application targets

run: ## Run the CLI with config.yml
	@if [ ! -f config.yml ]; then \
		echo "$(RED)Error: config.yml not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Running jira-agile-metrics...$(NC)"
	$(VENV_BIN)/python -m jira_agile_metrics.cli -vv config.yml

run-docker: ## Run the CLI using Docker (via run.sh)
	@echo "$(GREEN)Running with Docker...$(NC)"
	./run.sh

webapp: ## Start the web application
	@echo "$(GREEN)Starting web application...$(NC)"
	$(VENV_BIN)/python -m jira_agile_metrics.webapp.app

webapp-docker: ## Start the web application with Docker
	@echo "$(GREEN)Starting web application with Docker...$(NC)"
	docker run -d --rm -p 8080:80 --name $(CONTAINER_NAME) $(DOCKER_IMAGE_WEBAPP):latest

webapp-stop: ## Stop the web application Docker container
	@echo "$(YELLOW)Stopping web application...$(NC)"
	docker stop $(CONTAINER_NAME) || true
	@echo "$(GREEN)Web application stopped$(NC)"

## Docker targets

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE_DEV):latest -f Dockerfile.develop .

docker-build-webapp: ## Build webapp Docker image
	@echo "$(GREEN)Building webapp Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE_WEBAPP):latest -f Dockerfile.webapp .

docker-build: docker-build-dev ## Build all Docker images
	@echo "$(GREEN)Docker build complete$(NC)"

docker-clean: ## Remove Docker images and containers
	@echo "$(YELLOW)Cleaning Docker images...$(NC)"
	docker rmi $(DOCKER_IMAGE_DEV):latest 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE_WEBAPP):latest 2>/dev/null || true
	@echo "$(GREEN)Docker cleanup complete$(NC)"

## CI/CD targets

check-full: format-check lint test ## Run all checks and tests (CI-friendly)
	@echo "$(GREEN)All checks and tests passed!$(NC)"

ci: check-full ## Alias for check-full (CI usage)
	@echo "$(GREEN)CI checks complete$(NC)"

## Development workflow targets

dev: install-dev ## Full development setup
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC)"
	@echo "  make format     # Format code"
	@echo "  make lint       # Run linters"
	@echo "  make test       # Run tests"
	@echo "  make run        # Run application"

reset: clean-venv ## Reset development environment (reinstall everything)
	@echo "$(YELLOW)Resetting development environment...$(NC)"
	$(MAKE) install-dev
	@echo "$(GREEN)Development environment reset complete$(NC)"

## Documentation targets

docs: ## Generate documentation (if applicable)
	@echo "$(YELLOW)No documentation generation configured$(NC)"

## Release targets

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	$(VENV_BIN)/python -m build

publish: build ## Publish to PyPI (requires credentials)
	@echo "$(GREEN)Publishing to PyPI...$(NC)"
	$(VENV_BIN)/python -m twine upload dist/*

## Info targets

info: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Virtual env: $(VENV)"
	@if [ -d $(VENV) ]; then \
		echo "  Venv exists: Yes"; \
	else \
		echo "  Venv exists: No"; \
	fi
	@echo "  Requirements: $$(wc -l < requirements.txt) lines"
	@echo "  Source files: $$(find jira_agile_metrics -name '*.py' | wc -l) files"

version: ## Show version information
	@echo "$(GREEN)Version Information:$(NC)"
	@if [ -f setup.py ]; then \
		grep -E "version\s*=" setup.py | head -1; \
	else \
		echo "  setup.py not found"; \
	fi
