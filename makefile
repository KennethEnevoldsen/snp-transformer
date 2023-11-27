
install:
	@echo "--- 📦 Installing dependencies ---\n"
	pip install -e ".[dev, docs, tests]"

static-type-check:
	@echo "--- 🔍 Running static type checks ---\n"
	pyright src/.

test:
	@echo "--- 🧪 Running tests ---\n"
	@echo "Arguments:"
	@echo "  -v: verbose output\n"
	@echo ""
	pytest -v

lint:
	@echo "--- 🔧 Running linters ---\n"
	pre-commit run --all-files

pr:
	@echo "--- 📦 Running pre-commit checks ---\n"
	make install
	make static-type-check
	make lint
	make test
