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

run: install index search_dataset evaluate_docs evaluate_code answer_dataset

improved: install
	ollama pull nomic-embed-text
	$(RUN) index --embedding && make search_dataset --hybrid && make evaluate_docs && make evaluate_code && make answer_dataset

install: data Makefile
	uv sync
	ollama pull qwen3:0.6b
	ollama pull nomic-embed-text
	@echo "\033[0;32m\n[OK] installation completed ✔\n\033[0m"

debug:
	uv run pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf moulinette_pkg
	rm -rf .venv
	rm -rf data

lint:
	flake8 src && mypy $(LINT_FLAG) src

lint-strict:
	flake8 src && mypy src --strict

data: data/.installed

data/.installed: zip/datasets_public.zip zip/vllm-0.10.1.zip
	mkdir -p data/raw
	unzip zip/datasets_public.zip -d data >/dev/null
	unzip zip/vllm-0.10.1.zip -d data/raw >/dev/null
	touch data/.installed

moulinette: moulinette_pkg/.installed

moulinette_pkg/.installed: zip/moulinette.zip
	unzip zip/moulinette.zip
	touch moulinette_pkg/.installed

index:
	$(RUN) index $(ARG)

search:
	$(RUN) search $(QUERY)

search_dataset:
	$(RUN) search_dataset $(PUBLIC_DOCS_DATASET) $(ARG)
	$(RUN) search_dataset $(PUBLIC_CODE_DATASET) $(ARG)

answer:
	$(RUN) answer $(QUERY)

answer_dataset:
	$(RUN) answer_dataset data/output/search_results/dataset_docs_public.json $(ARG)
	$(RUN) answer_dataset data/output/search_results/dataset_code_public.json $(ARG)

evaluate_docs: moulinette
	./moulinette_pkg/moulinette-ubuntu evaluate_student_search_results $(PUBLIC_DOCS_SEARCH_RESULTS) $(PUBLIC_DOCS_ANSWERED_QUESTIONS)

evaluate_code: moulinette
	./moulinette_pkg/moulinette-ubuntu evaluate_student_search_results $(PUBLIC_CODE_SEARCH_RESULTS) $(PUBLIC_CODE_ANSWERED_QUESTIONS)

recall_docs:
	$(RUN) evaluate $(PUBLIC_DOCS_SEARCH_RESULTS) $(PUBLIC_DOCS_ANSWERED_QUESTIONS)

recall_code:
	$(RUN) evaluate $(PUBLIC_CODE_SEARCH_RESULTS) $(PUBLIC_CODE_ANSWERED_QUESTIONS)

.PHONY: run install debug clean lint lint-strict data moulinette index search search_dataset answer answer_dataset evaluate recall_docs recall_code

best:
	make index ARG="--max_chunk_size=1200" && make search_dataset && make evaluate_docs && make evaluate_code

exam: moulinette
	cp ~/Downloads/exams.zip . 
	cp ~/Downloads/datasets_private.zip .
	mkdir -p data/datasets
	unzip exams.zip
	unzip datasets_private.zip -d data/datasets
	./exams/scripts/exam_retrieval.sh --student-path . --moulinette-path ./moulinette_pkg/moulinette-ubuntu
# 	./exams/scripts/exam_answer.sh --student-path . --moulinette-path ./moulinette_pkg/moulinette-ubuntu