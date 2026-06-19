ARG ?=
RUN := uv run -m src
LINT_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

run: install
	$(RUN) $(ARG)

install:
	uv sync
	vllm

debug:
	uv run pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .venv
	rm -rf data/raw/vllm-0.10.1
	rm -rf data/processed

lint:
	flake8 && mypy $(LINT_FLAG) src

lint-strict:
	mypy src && flake8 src --strict

vllm: data/raw/vllm-0.10.1/.installed

data/raw/vllm-0.10.1/.installed: zip/vllm-0.10.1.zip
	mkdir -p data/raw
	unzip zip/vllm-0.10.1.zip -d data/raw
	touch data/raw/vllm-0.10.1/.installed

moulinette:
	unzip zip/moulinete.zip

.PHONY: run install debug clean lint lint-strict vllm index search search_dataset answer answer_dataset evaluate


# index:
# 	$(RUN) index $(ARG)
# uv run python -m src index --max_chunk_size 2000
# search:
# 	$(RUN) search "How to configure OpenAI server?"
# uv run python -m student search --k 10
# search_dataset:
# 	$(RUN) search_dataset
# answer: search
# 	$(RUN) answer $(ARG)
# uv run python -m student answer "How to configure OpenAI server?" --k 10
# answer_dataset: search_dataset
# 	$(RUN) answer_dataset
# evaluate:
# 	$(RUN) evaluate
