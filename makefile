
static-type-check:
	pyright src/.

test:
	# Run pytest with 
	# verbose output (-v) and 
	# automatically determine the number of worker processes to use (-n auto)
	pytest -v -n auto

lint:
	pre-commit run --all-files

pretrain:
	python src/pretrain/pretrain.py
