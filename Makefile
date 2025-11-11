# Jira Agile Metrics - Makefile
# Provides common development and deployment tasks

.PHONY: help install install-dev clean test test-functional test-e2e lint lint-ruff lint-makefile validate-makefile format format-code format-markdown format-yaml format-toml format-makefile format-check format-code-check format-markdown-check format-yaml-check format-toml-check format-makefile-check check check-toml docker-build docker-run webapp run check-full test-all lint-fix test-coverage pylint \
	docker-test docker-test-functional docker-test-e2e docker-test-all docker-lint docker-format docker-check \
	docker-cli docker-cli-verbose docker-compose-up docker-compose-down docker-build-prod

# Python interpreter
PYTHON = python3
VENV = .venv
VENV_BIN = $(VENV)/bin
VENV_PYTHON = $(VENV_BIN)/python3
PIP = $(VENV_BIN)/pip
PYTEST = $(VENV_BIN)/pytest
PTW = $(VENV_BIN)/ptw
BLACK = $(VENV_BIN)/black
RUFF = $(VENV_BIN)/ruff
PYLINT = $(VENV_BIN)/pylint
MYPY = $(VENV_BIN)/mypy
YAMLLINT = $(VENV_BIN)/yamllint
PYMARKDOWN = $(VENV_BIN)/pymarkdown
YAMLFIX = $(VENV_BIN)/yamlfix
MDFORMAT = $(VENV_BIN)/mdformat
TOMLSORT = $(VENV_BIN)/toml-sort
MBAKE = $(VENV_BIN)/mbake
CHECK_MAKEFILE = $(VENV_BIN)/check-makefile

# Common paths for linting/formatting
# Include tests explicitly so helpers like tests/helpers/csv_utils.py are linted
LINT_PATHS = jira_agile_metrics/ jira_agile_metrics/tests/ setup.py

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
	@echo "  make docker-test  # Run tests inside Docker"
	@echo "  make docker-lint  # Run linters inside Docker"
	@echo "  make docker-cli   # Run CLI via Docker"

## Development targets

venv: ## Create virtual environment
	@if [ ! -d $(VENV) ]; then \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
	fi

install: venv ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-prod.txt

install-dev: venv ## Install all development dependencies (includes production + dev tooling)
	@echo "$(GREEN)Installing all development dependencies...$(NC)"
	@echo "$(YELLOW)This includes production dependencies and all dev tooling (linting, formatting, etc.)$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

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

test: ## Run unit tests (excludes functional and e2e tests)
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) -v -m "not e2e and not functional"

test-functional: ## Run functional tests (CSV E2E)
	@echo "$(GREEN)Running functional tests...$(NC)"
	$(PYTEST) -v -m functional

test-e2e: ## Run end-to-end tests (full application flow)
	@echo "$(GREEN)Running end-to-end tests...$(NC)"
	$(PYTEST) -v -m e2e

test-coverage: ## Run tests with coverage (excludes functional and e2e tests)
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) --cov=jira_agile_metrics --cov-report=html --cov-report=term -m "not e2e and not functional"

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests with verbose output...$(NC)"
	$(PYTEST) -vv

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(PTW)

test-all: ## Run all tests (unit, functional, and e2e)
	@echo "$(GREEN)Running all tests...$(NC)"
	$(MAKE) test && $(MAKE) test-functional && $(MAKE) test-e2e

## Linting and formatting targets

lint: lint-ruff pylint lint-markdown lint-yaml lint-makefile validate-makefile check-toml ## Run all linters in check-only mode (ruff, pylint, markdown, yaml, makefile validation and reference checking, toml) - does not modify files

lint-ruff: ## Run ruff in check-only mode (does not modify files)
	@echo "$(GREEN)Running ruff check (no auto-fix)...$(NC)"
	$(RUFF) check $(LINT_PATHS)

lint-fix: ## Run ruff with auto-fix (modifies source files)
	@echo "$(GREEN)Running ruff with auto-fix...$(NC)"
	$(RUFF) check $(LINT_PATHS) --fix

pylint: ## Run pylint
	@echo "$(GREEN)Running pylint on codebase...$(NC)"
	$(PYLINT) $(LINT_PATHS)

format: format-code format-markdown format-yaml format-toml format-makefile ## Format all code (Python, Markdown, YAML, TOML, Makefile)

format-code: ## Format Python code with ruff
	@echo "$(GREEN)Formatting Python code with ruff...$(NC)"
	$(RUFF) format $(LINT_PATHS)

format-check: format-code-check format-markdown-check format-yaml-check format-toml-check format-makefile-check ## Check all formatting without making changes
	@echo "$(GREEN)All format checks passed!$(NC)"

format-code-check: ## Check Python code formatting without making changes
	@echo "$(GREEN)Checking Python code formatting...$(NC)"
	$(RUFF) format $(LINT_PATHS) --check

format-markdown: ## Format markdown files (includes .cursor directory)
	@echo "$(GREEN)Formatting markdown files...$(NC)"
	@FILES=$$(find . -name "*.md" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./htmlcov/*" -not -path "./build/*" -not -path "./dist/*" -not -path "./node_modules/*" 2>/dev/null); \
	if [ -n "$$FILES" ]; then \
		if [ -f "$(MDFORMAT)" ]; then \
			echo "$$FILES" | xargs $(MDFORMAT) 2>/dev/null || true; \
		elif command -v mdformat >/dev/null 2>&1; then \
			echo "$$FILES" | xargs mdformat 2>/dev/null || true; \
		else \
			echo "$(YELLOW)mdformat not found. Run 'make install-dev' to install tools.$(NC)"; \
		fi; \
	fi

format-markdown-check: ## Check markdown formatting without making changes (includes .cursor directory)
	@echo "$(GREEN)Checking markdown formatting...$(NC)"
	@FILES=$$(find . -name "*.md" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./htmlcov/*" -not -path "./build/*" -not -path "./dist/*" -not -path "./node_modules/*" 2>/dev/null); \
	if [ -n "$$FILES" ]; then \
		if [ -f "$(MDFORMAT)" ]; then \
			echo "$$FILES" | xargs $(MDFORMAT) --check 2>/dev/null || true; \
		elif command -v mdformat >/dev/null 2>&1; then \
			echo "$$FILES" | xargs mdformat --check 2>/dev/null || true; \
		fi; \
	fi

format-yaml: ## Format YAML files
	@echo "$(GREEN)Formatting YAML files...$(NC)"
	@FILES=$$(find . \( -name "*.yml" -o -name "*.yaml" \) -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./htmlcov/*" -not -path "./build/*" -not -path "./dist/*" -not -path "./node_modules/*" 2>/dev/null); \
	if [ -n "$$FILES" ]; then \
		if [ -f "$(YAMLFIX)" ]; then \
			echo "$$FILES" | xargs $(YAMLFIX) --config-file .yamllint.yml 2>/dev/null || true; \
		elif command -v yamlfix >/dev/null 2>&1; then \
			echo "$$FILES" | xargs yamlfix --config-file .yamllint.yml 2>/dev/null || true; \
		else \
			echo "$(YELLOW)yamlfix not found. Run 'make install-dev' to install tools.$(NC)"; \
		fi; \
	fi

format-yaml-check: ## Check YAML formatting without making changes
	@echo "$(GREEN)Checking YAML formatting...$(NC)"
	@FILES=$$(find . \( -name "*.yml" -o -name "*.yaml" \) -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./htmlcov/*" -not -path "./build/*" -not -path "./dist/*" -not -path "./node_modules/*" 2>/dev/null); \
	if [ -n "$$FILES" ]; then \
		if [ -f "$(YAMLFIX)" ]; then \
			echo "$$FILES" | xargs $(YAMLFIX) --check --config-file .yamllint.yml 2>/dev/null || true; \
		elif command -v yamlfix >/dev/null 2>&1; then \
			echo "$$FILES" | xargs yamlfix --check --config-file .yamllint.yml 2>/dev/null || true; \
		fi; \
	fi

format-toml: ## Format TOML files
	@echo "$(GREEN)Formatting TOML files...$(NC)"
	@if [ -f "$(TOMLSORT)" ]; then \
		$(TOMLSORT) --in-place pyproject.toml mise.toml || true; \
	elif command -v toml-sort >/dev/null 2>&1; then \
		toml-sort --in-place pyproject.toml mise.toml || true; \
	else \
		echo "$(YELLOW)toml-sort not found. Run 'make install-dev' to install tools.$(NC)"; \
	fi

format-toml-check: ## Check TOML formatting without making changes
	@echo "$(GREEN)Checking TOML formatting...$(NC)"
	@if [ -f "$(TOMLSORT)" ]; then \
		$(TOMLSORT) --check pyproject.toml mise.toml || true; \
	elif command -v toml-sort >/dev/null 2>&1; then \
		toml-sort --check pyproject.toml mise.toml || true; \
	fi

format-makefile: ## Format Makefile
	@echo "$(GREEN)Formatting Makefile...$(NC)"
	@if [ -f "$(MBAKE)" ]; then \
		$(MBAKE) format Makefile || true; \
	elif command -v mbake >/dev/null 2>&1; then \
		mbake format Makefile || true; \
	else \
		echo "$(YELLOW)mbake not found. Run 'make install-dev' to install tools.$(NC)"; \
	fi

format-makefile-check: ## Check Makefile formatting without making changes
	@echo "$(GREEN)Checking Makefile formatting...$(NC)"
	@if [ -f "$(MBAKE)" ]; then \
		$(MBAKE) format --check Makefile || true; \
	elif command -v mbake >/dev/null 2>&1; then \
		mbake format --check Makefile || true; \
	fi

lint-markdown: ## Lint markdown files (includes .cursor directory)
	@echo "$(GREEN)Linting markdown files...$(NC)"
	@FILES=$$(find . -name "*.md" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./htmlcov/*" -not -path "./build/*" -not -path "./dist/*" -not -path "./node_modules/*" 2>/dev/null); \
	if [ -n "$$FILES" ]; then \
		if [ -f "$(PYMARKDOWN)" ]; then \
			echo "$$FILES" | xargs $(PYMARKDOWN) scan \
			--set "plugins.md013.line_length=120" \
			--set "plugins.md013.tables=false" \
			--set "plugins.md013.code_blocks=false" \
			2>/dev/null || true; \
		elif command -v pymarkdown >/dev/null 2>&1; then \
			echo "$$FILES" | xargs pymarkdown scan \
			--set "plugins.md013.line_length=120" \
			--set "plugins.md013.tables=false" \
			--set "plugins.md013.code_blocks=false" \
			2>/dev/null || true; \
		fi; \
	fi

lint-yaml: ## Lint YAML files (uses .yamllint.yml config)
	@echo "$(GREEN)Linting YAML files...$(NC)"
	@FILES=$$(find . \( -name "*.yml" -o -name "*.yaml" \) -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./htmlcov/*" -not -path "./build/*" -not -path "./dist/*" -not -path "./node_modules/*" 2>/dev/null); \
	if [ -n "$$FILES" ]; then \
		echo "$$FILES" | xargs $(YAMLLINT) -c .yamllint.yml 2>/dev/null || true; \
	fi

lint-makefile: ## Lint Makefile syntax
	@echo "$(GREEN)Linting Makefile...$(NC)"
	@if [ -f "$(MBAKE)" ]; then \
		$(MBAKE) validate Makefile || true; \
	elif command -v mbake >/dev/null 2>&1; then \
		mbake validate Makefile || true; \
	else \
		echo "$(YELLOW)mbake not found. Run 'make install-dev' to install tools.$(NC)"; \
	fi

validate-makefile: ## Validate Makefile (check that referenced scripts/files exist)
	@echo "$(GREEN)Validating Makefile references...$(NC)"
	@if [ -f "scripts/check_makefile.py" ]; then \
		$(VENV_PYTHON) scripts/check_makefile.py || true; \
	elif [ -f "$(CHECK_MAKEFILE)" ]; then \
		$(CHECK_MAKEFILE) || true; \
	elif command -v check-makefile >/dev/null 2>&1; then \
		check-makefile || true; \
	else \
		echo "$(YELLOW)makefile-checker not available. Run 'make install-dev' to install tools.$(NC)"; \
	fi

check-toml: ## Check TOML formatting (format consistency check)
	@echo "$(GREEN)Checking TOML formatting...$(NC)"
	@$(MAKE) format-toml-check

check: format-check lint ## Run all checks without making changes (format + lint)
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

# Additional Docker helpers for portability

docker-build-prod: ## Build production CLI Docker image
	@echo "$(GREEN)Building production Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE_PROD):latest -f Dockerfile .

docker-test: docker-build-dev ## Run unit tests inside Docker
	@echo "$(GREEN)Running unit tests in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE_DEV):latest pytest -v -m "not e2e and not functional"

docker-test-functional: docker-build-dev ## Run functional tests inside Docker
	@echo "$(GREEN)Running functional tests in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE_DEV):latest pytest -v -m functional

docker-test-e2e: docker-build-dev ## Run e2e tests inside Docker
	@echo "$(GREEN)Running e2e tests in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE_DEV):latest pytest -v -m e2e

docker-test-all: ## Run unit, functional, and e2e tests inside Docker
	@echo "$(GREEN)Running all tests in Docker...$(NC)"
	$(MAKE) docker-test && $(MAKE) docker-test-functional && $(MAKE) docker-test-e2e

docker-lint: docker-build-dev ## Run linters (ruff, pylint, markdown, yaml, toml) inside Docker
	@echo "$(GREEN)Running linters in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE_DEV):latest sh -c "ruff check $(LINT_PATHS) && pylint $(LINT_PATHS) && (make lint-yaml && make lint-markdown && make check-toml) || true"

docker-format: docker-build-dev ## Format code with black inside Docker
	@echo "$(GREEN)Formatting with black in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE_DEV):latest black $(LINT_PATHS)

docker-check: docker-build-dev ## Run format-check and linters inside Docker
	@echo "$(GREEN)Checking formatting and lint in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE_DEV):latest sh -c "black $(LINT_PATHS) --check && ruff check $(LINT_PATHS) && pylint $(LINT_PATHS)"

docker-cli: docker-build-prod ## Run CLI with config.yml via Docker
	@echo "$(GREEN)Running CLI via Docker...$(NC)"
	@if [ ! -f config.yml ]; then echo "$(RED)Error: config.yml not found$(NC)"; exit 1; fi
	@if [ -f .env ]; then \
		docker run --rm -v $(PWD):/data --env-file .env -e MPLBACKEND=agg $(DOCKER_IMAGE_PROD):latest -vv /data/config.yml; \
	else \
		docker run --rm -v $(PWD):/data -e MPLBACKEND=agg $(DOCKER_IMAGE_PROD):latest -vv /data/config.yml; \
	fi

docker-cli-verbose: docker-build-prod ## Run CLI with extra verbose via Docker
	@echo "$(GREEN)Running CLI via Docker (very verbose)...$(NC)"
	@if [ ! -f config.yml ]; then echo "$(RED)Error: config.yml not found$(NC)"; exit 1; fi
	@if [ -f .env ]; then \
		docker run --rm -v $(PWD):/data --env-file .env -e MPLBACKEND=agg $(DOCKER_IMAGE_PROD):latest -vvv /data/config.yml; \
	else \
		docker run --rm -v $(PWD):/data -e MPLBACKEND=agg $(DOCKER_IMAGE_PROD):latest -vvv /data/config.yml; \
	fi

docker-compose-up: ## Start webapp via docker-compose
	@echo "$(GREEN)Starting webapp with docker-compose...$(NC)"
	docker compose up -d

docker-compose-down: ## Stop webapp via docker-compose
	@echo "$(YELLOW)Stopping webapp with docker-compose...$(NC)"
	docker compose down

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