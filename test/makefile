# Behavioral Boredom Index - Development Makefile
# Professional development workflow automation

.PHONY: help install install-dev setup clean test lint format type-check security
.PHONY: docs serve-docs build docker-build docker-run deploy performance
.PHONY: data-setup federated-setup benchmark profile coverage

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit
DOCKER := docker
PROJECT_NAME := behavioral-boredom-index
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Behavioral Boredom Index - Development Commands$(NC)"
	@echo "=================================================="
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)Quick Start:$(NC)"
	@echo "  make install-dev  # Install development dependencies"
	@echo "  make setup        # Complete development setup"
	@echo "  make test         # Run test suite"
	@echo "  make lint         # Run code quality checks"

# Installation and Setup
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PYTHON) -m spacy download en_core_web_sm
	pre-commit install

setup: install-dev ## Complete development environment setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	mkdir -p logs data models config
	cp .env.example .env
	@echo "$(YELLOW)Don't forget to configure your .env file!$(NC)"
	$(PYTHON) setup.py develop
	@echo "$(GREEN)Setup complete! ✅$(NC)"

clean: ## Clean up temporary files and caches
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.coverage" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf logs/*.log
	$(DOCKER) system prune -f

# Testing
test: ## Run the full test suite
	@echo "$(GREEN)Running test suite...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=80

test-fast: ## Run tests without coverage (faster)
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short -x

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/integration/ -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/unit/ -v --tb=short

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/performance/ -v --benchmark-only

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -f --tb=short

# Code Quality
lint: ## Run all linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	$(FLAKE8) $(SRC_DIR)/ $(TEST_DIR)/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)Linting passed! ✅$(NC)"

format: ## Format code with Black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	$(BLACK) $(SRC_DIR)/ $(TEST_DIR)/
	isort $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)Code formatted! ✅$(NC)"

format-check: ## Check if code is properly formatted
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(BLACK) --check --diff $(SRC_DIR)/ $(TEST_DIR)/
	isort --check-only --diff $(SRC_DIR)/ $(TEST_DIR)/

type-check: ## Run type checking with MyPy
	@echo "$(GREEN)Running type checks...$(NC)"
	$(MYPY) $(SRC_DIR)/ --ignore-missing-imports
	@echo "$(GREEN)Type checking passed! ✅$(NC)"

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	$(BANDIT) -r $(SRC_DIR)/ -f json -o security-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)Security checks completed! ✅$(NC)"

quality: format lint type-check security ## Run all code quality checks
	@echo "$(GREEN)All quality checks passed! ✅$(NC)"

# Documentation
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd $(DOCS_DIR) && make html
	@echo "$(GREEN)Documentation built! Check docs/_build/html/index.html$(NC)"

docs-clean: ## Clean documentation build
	@echo "$(GREEN)Cleaning documentation...$(NC)"
	cd $(DOCS_DIR) && make clean

serve-docs: docs ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8080$(NC)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8080

docs-live: ## Build and serve docs with live reload
	@echo "$(GREEN)Starting live documentation server...$(NC)"
	sphinx-autobuild $(DOCS_DIR) $(DOCS_DIR)/_build/html --host 0.0.0.0 --port 8080

# Building and Packaging
build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	twine check dist/*
	@echo "$(GREEN)Build completed! ✅$(NC)"

upload-test: build ## Upload to Test PyPI
	@echo "$(GREEN)Uploading to Test PyPI...$(NC)"
	twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	@echo "$(GREEN)Uploading to PyPI...$(NC)"
	twine upload dist/*

# Docker
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):latest .
	$(DOCKER) build -t $(PROJECT_NAME):dashboard --target dashboard .

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):dev --target development .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	$(DOCKER) run -p 8000:8000 $(PROJECT_NAME):latest

docker-run-dashboard: ## Run Dashboard Docker container
	@echo "$(GREEN)Running Dashboard Docker container...$(NC)"
	$(DOCKER) run -p 8501:8501 $(PROJECT_NAME):dashboard

docker-compose-up: ## Start all services with docker-compose
	@echo "$(GREEN)Starting all services...$(NC)"
	docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "$(GREEN)Stopping all services...$(NC)"
	docker-compose down

# Application Running
run-api: ## Run the API server
	@echo "$(GREEN)Starting API server...$(NC)"
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

run-dashboard: ## Run the Streamlit dashboard
	@echo "$(GREEN)Starting dashboard...$(NC)"
	streamlit run dashboard.py --server.port 8501

run-dev: ## Run in development mode with auto-reload
	@echo "$(GREEN)Starting development server...$(NC)"
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug

run-worker: ## Run background worker
	@echo "$(GREEN)Starting background worker...$(NC)"
	$(PYTHON) -m bbi.worker

# Data and ML Operations
data-setup: ## Set up sample data for development
	@echo "$(GREEN)Setting up sample data...$(NC)"
	$(PYTHON) scripts/generate_sample_data.py
	@echo "$(GREEN)Sample data generated! ✅$(NC)"

federated-setup: ## Set up federated learning environment
	@echo "$(GREEN)Setting up federated learning...$(NC)"
	$(PYTHON) scripts/setup_federated_learning.py
	@echo "$(GREEN)Federated learning setup complete! ✅$(NC)"

train-model: ## Train the ML model
	@echo "$(GREEN)Training ML model...$(NC)"
	$(PYTHON) -m bbi.training.train_model

evaluate-model: ## Evaluate model performance
	@echo "$(GREEN)Evaluating model...$(NC)"
	$(PYTHON) -m bbi.evaluation.evaluate_model

# Performance and Profiling
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTEST) $(TEST_DIR)/performance/ --benchmark-only --benchmark-json=benchmark.json
	@echo "$(GREEN)Benchmarks completed! Check benchmark.json$(NC)"

profile: ## Profile application performance
	@echo "$(GREEN)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats -m bbi.cli analyze --sample-data
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile: ## Run memory profiling
	@echo "$(GREEN)Running memory profiling...$(NC)"
	mprof run $(PYTHON) -m bbi.cli analyze --sample-data
	mprof plot

load-test: ## Run load testing
	@echo "$(GREEN)Running load tests...$(NC)"
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Coverage and Reporting
coverage: ## Generate detailed coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	$(PYTEST) $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=xml
	@echo "$(GREEN)Coverage report generated! Check htmlcov/index.html$(NC)"

coverage-badge: coverage ## Generate coverage badge
	@echo "$(GREEN)Generating coverage badge...$(NC)"
	coverage-badge -o coverage.svg

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(NC)"
	# Add your staging deployment commands here
	@echo "$(YELLOW)Staging deployment not configured$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(GREEN)Deploying to production...$(NC)"
	# Add your production deployment commands here
	@echo "$(YELLOW)Production deployment not configured$(NC)"

# Database Operations
db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	alembic upgrade head

db-reset: ## Reset database
	@echo "$(RED)Resetting database...$(NC)"
	$(PYTHON) scripts/reset_database.py

db-seed: ## Seed database with sample data
	@echo "$(GREEN)Seeding database...$(NC)"
	$(PYTHON) scripts/seed_database.py

# Monitoring and Health
health-check: ## Check application health
	@echo "$(GREEN)Checking application health...$(NC)"
	curl -f http://localhost:8000/health || echo "$(RED)API not running$(NC)"
	curl -f http://localhost:8501 || echo "$(RED)Dashboard not running$(NC)"

logs: ## Show application logs
	@echo "$(GREEN)Showing recent logs...$(NC)"
	tail -f logs/bbi.log

monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Starting monitoring...$(NC)"
	$(PYTHON) scripts/monitor.py

# Utilities
check-deps: ## Check for dependency updates
	@echo "$(GREEN)Checking for dependency updates...$(NC)"
	pip list --outdated

update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	pip-review --auto

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

pre-push: quality test ## Run all checks before pushing
	@echo "$(GREEN)All pre-push checks passed! Ready to push! 🚀$(NC)"

# Development workflow shortcuts
dev: install-dev setup ## Quick development setup
	@echo "$(GREEN)Development environment ready! 🎯$(NC)"

ci: quality test ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed successfully! ✅$(NC)"

release: quality test build ## Prepare for release
	@echo "$(GREEN)Release preparation completed! 🚀$(NC)"

# Help for specific categories
help-test: ## Show testing commands
	@echo "$(BLUE)Testing Commands:$(NC)"
	@echo "  test           - Run full test suite"
	@echo "  test-fast      - Run tests without coverage"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-performance - Run performance tests"

help-quality: ## Show code quality commands
	@echo "$(BLUE)Code Quality Commands:$(NC)"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code"
	@echo "  type-check     - Run type checking"
	@echo "  security       - Run security checks"
	@echo "  quality        - Run all quality checks"

help-docker: ## Show Docker commands
	@echo "$(BLUE)Docker Commands:$(NC)"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-compose-up - Start all services"
