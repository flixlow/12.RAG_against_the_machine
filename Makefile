ARG ?=
RUN := uv run -m src
QUERY ?= 'How to configure OpenAI server?'
LINT_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

PUBLIC_DOCS_DATASET := data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json
PUBLIC_CODE_DATASET := data/datasets_public/public/UnansweredQuestions/dataset_code_public.json
PUBLIC_DOCS_ANSWERED_QUESTIONS := data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json
PUBLIC_CODE_ANSWERED_QUESTIONS := data/datasets_public/public/AnsweredQuestions/dataset_code_public.json
PUBLIC_DOCS_SEARCH_RESULTS := data/output/search_results/dataset_docs_public.json
PUBLIC_CODE_SEARCH_RESULTS := data/output/search_results/dataset_code_public.json

install: data Makefile
	uv sync
	ollama pull qwen3:0.6b
	@echo "\033[0;32m\n[OK] installation completed ✔\n\033[0m"

run: install index search_dataset evaluate_docs evaluate_code answer_dataset

debug:
	uv run pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf moulinette-ubuntu
	rm -rf moulinette-fedora
	rm -rf .moulinette
	rm -rf .venv
	rm -rf data

lint:
	uv run -m flake8 src && uv run -m mypy $(LINT_FLAG) src

index:
	$(RUN) index $(ARG)

search:
	$(RUN) search $(QUERY) $(ARG)

search_dataset:
	$(RUN) search_dataset $(PUBLIC_DOCS_DATASET) $(ARG)
	$(RUN) search_dataset $(PUBLIC_CODE_DATASET) $(ARG)

answer:
	$(RUN) answer $(QUERY) $(ARG)

answer_dataset:
	$(RUN) answer_dataset data/output/search_results/dataset_docs_public.json $(ARG)
	$(RUN) answer_dataset data/output/search_results/dataset_code_public.json $(ARG)

evaluate_docs: moulinette
	./moulinette-ubuntu evaluate_student_search_results $(PUBLIC_DOCS_SEARCH_RESULTS) $(PUBLIC_DOCS_ANSWERED_QUESTIONS)

evaluate_code: moulinette
	./moulinette-ubuntu evaluate_student_search_results $(PUBLIC_CODE_SEARCH_RESULTS) $(PUBLIC_CODE_ANSWERED_QUESTIONS)

recall_docs:
	$(RUN) evaluate $(PUBLIC_DOCS_SEARCH_RESULTS) $(PUBLIC_DOCS_ANSWERED_QUESTIONS)

recall_code:
	$(RUN) evaluate $(PUBLIC_CODE_SEARCH_RESULTS) $(PUBLIC_CODE_ANSWERED_QUESTIONS)

api:
	uv run -m fastapi dev src/api.py

data: data/.installed

data/.installed: datasets_public.zip vllm-0.10.1.zip
	mkdir -p data/raw
	unzip datasets_public.zip -d data >/dev/null
	unzip vllm-0.10.1.zip -d data/raw >/dev/null
	touch data/.installed

moulinette: .moulinette

.moulinette: moulinette.zip
	unzip moulinette.zip
	touch .moulinette

.PHONY: run install debug clean lint data moulinette index search search_dataset answer answer_dataset evaluate_docs evaluate_code recall_docs recall_code
