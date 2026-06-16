LINT_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

run: install
	uv run -m src

install:
	uv sync

debug:
	uv run pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .venv
	rm -rf vllm-0.10.1

lint:
	flake8 && mypy $(LINT_FLAG) src

lint-strict:
	mypy src && flake8 src --strict

vllm: vllm-0.10.1/.installed

vllm-0.10.1/.installed: data/vllm-0.10.1.zip
	unzip data/vllm-0.10.1.zip
	touch vllm-0.10.1/.installed

.PHONY: run install debug clean lint lint-strict vllm