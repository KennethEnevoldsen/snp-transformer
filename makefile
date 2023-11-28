
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

pretrain:
	python src/pretrain/pretrain.py


install_on_ucloud:
	# does not work with just:
	pip install -e ".[tests, docs, dev]"
	# get import error (seems like an error of installation)

	# followed: https://github.com/pytorch/pytorch/issues/111469
	# and tried:
	python -m pip uninstall torch torchvision torchaudio
	python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
	# solves it but leads to an issue with pytorch lightning
	# works with import torch, but not pl

	# try:
	python -m pip install torch==2.0.1
	# seems to work, but now I have problems with torchmetrics (probably due to torchvision not matching)
	# uninstall it seems to do the trick!

lint:
	@echo "--- 🔧 Running linters ---\n"
	pre-commit run --all-files

pr:
	@echo "--- 📦 Running pre-commit checks ---\n"
	make install
	make static-type-check
	make lint
	make test
