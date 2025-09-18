.PHONY: all format lint test test_integration lint_libs lint_apps format_libs format_apps test_libs test_apps test_integration_libs test_integration_apps help

# Default target executed when no arguments are given to make.
all: help

######################
# SHARED FUNCTIONS
######################

# Function to run tests for a package type
define run_tests
	@echo "Testing $(1) packages..."
	@failed=0; \
	for dir in $(1)/*/; do \
		if [ -d "$$dir/tests" ]; then \
			echo "Testing $$dir..."; \
			cd "$$dir" && uv run python -m pytest tests/unit/ || failed=1; \
			cd - >/dev/null; \
		fi \
	done; \
	if [ $$failed -eq 1 ]; then \
		echo "❌ Some tests failed in $(1) packages"; \
		exit 1; \
	else \
		echo "✅ All tests passed in $(1) packages"; \
	fi
endef

# Function to run integration tests for a package type
define run_integration_tests
	@echo "Running integration tests for $(1) packages..."
	@failed=0; \
	for dir in $(1)/*/; do \
		if [ -d "$$dir/tests/integration" ]; then \
			echo "Integration testing $$dir..."; \
			cd "$$dir" && uv run python -m pytest tests/integration/ || failed=1; \
			cd - >/dev/null; \
		fi \
	done; \
	if [ $$failed -eq 1 ]; then \
		echo "❌ Some integration tests failed in $(1) packages"; \
		exit 1; \
	else \
		echo "✅ All integration tests passed in $(1) packages"; \
	fi
endef

# Function to run linting for a package type
define run_lint
	@echo "Linting $(1) packages..."
	@for dir in $(1)/*/; do \
		if [ -d "$$dir/src" ]; then \
			echo "Linting $$dir..."; \
			cd "$$dir" && \
			uv run ruff check . && \
			uv run ruff format src --diff && \
			uv run ruff check --select I src && \
			mkdir -p .mypy_cache && uv run mypy --strict src --cache-dir .mypy_cache; \
			cd - >/dev/null; \
		fi \
	done
endef

# Function to run formatting for a package type
define run_format
	@echo "Formatting $(1) packages..."
	@for dir in $(1)/*/; do \
		if [ -d "$$dir/src" ]; then \
			echo "Formatting $$dir..."; \
			cd "$$dir" && \
			uv run ruff format src && \
			uv run ruff check --select I --fix src; \
			cd - >/dev/null; \
		fi \
	done
endef

######################
# TESTING
######################

# Run all unit tests in monorepo
test: test_libs test_apps

# Run all tests in monorepo (unit + integration)
test_all: test_libs test_apps test_integration_libs test_integration_apps

# Run all integration tests in monorepo
test_integration: test_integration_libs test_integration_apps

# Test specific package types
test_libs:
	$(call run_tests,libs)

test_apps:
	$(call run_tests,apps)

# Integration test specific package types
test_integration_libs:
	$(call run_integration_tests,libs)

test_integration_apps:
	$(call run_integration_tests,apps)

######################
# LINTING AND FORMATTING
######################

# Lint all packages
lint: lint_libs lint_apps

# Format all packages
format: format_libs format_apps

# Lint specific package types
lint_libs:
	$(call run_lint,libs)

lint_apps:
	$(call run_lint,apps)

# Format specific package types
format_libs:
	$(call run_format,libs)

format_apps:
	$(call run_format,apps)


######################
# HELP
######################

help:
	@echo '----'
	@echo 'MONOREPO COMMANDS:'
	@echo 'test                         - run all unit tests (libs + apps)'
	@echo 'test_all                     - run all tests: unit + integration (libs + apps)'
	@echo 'test_integration             - run all integration tests (libs + apps)'
	@echo 'lint                         - run linters on all packages (libs + apps)'
	@echo 'format                       - run formatters on all packages (libs + apps)'
	@echo ''
	@echo 'PACKAGE TYPE TARGETS:'
	@echo 'test_libs                    - test libs/ packages only (unit tests)'
	@echo 'test_apps                    - test apps/ packages only (unit tests)'
	@echo 'test_integration_libs        - integration test libs/ packages only'
	@echo 'test_integration_apps        - integration test apps/ packages only'
	@echo 'lint_libs                    - lint libs/ packages only'
	@echo 'lint_apps                    - lint apps/ packages only'
	@echo 'format_libs                  - format libs/ packages only'
	@echo 'format_apps                  - format apps/ packages only'