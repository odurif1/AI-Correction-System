.PHONY: help scan audit lint test dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

scan: ## Run security scans (pip-audit + bandit)
	@./scripts/security-scan.sh

audit: ## Run pip-audit only
	@echo "=== Dependency Vulnerability Scan ==="
	@pip-audit --desc

lint: ## Run bandit only
	@echo "=== Static Security Analysis ==="
	@bandit -r src/ --severity-level medium --confidence-level medium

test: ## Run pytest tests
	@python -m pytest tests/ -v

dev: ## Start development server
	@uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
