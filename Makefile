ARG ?=
RUN := uv run -m src
LINT_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

PUBLIC_DOCS_ANSWERED_QUESTIONS := data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json
PUBLIC_CODE_ANSWERED_QUESTIONS := data/datasets_public/public/AnsweredQuestions/dataset_code_public.json
PUBLIC_DOCS_SEARCH_RESULTS := data/output/search_results/dataset_docs_public.json
PUBLIC_CODE_SEARCH_RESULTS := data/output/search_results/dataset_code_public.json

run: install index

install: data
	uv sync

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
	unzip zip/datasets_public.zip -d data
	unzip zip/vllm-0.10.1.zip -d data/raw
	touch data/.installed

moulinette: moulinette_pkg/.installed

moulinette_pkg/.installed: zip/moulinette.zip
	unzip zip/moulinette.zip
	touch moulinette_pkg/.installed

index:
	$(RUN) index $(ARG)

search:
	$(RUN) search $(ARG)
# make search ARG="'How to configure OpenAI server?'"

search_dataset:
	$(RUN) search_dataset $(ARG)

answer:
	$(RUN) answer $(ARG)

answer_dataset:
	$(RUN) answer_dataset

evaluate_docs: moulinette
	./moulinette_pkg/moulinette-ubuntu evaluate_student_search_results $(PUBLIC_DOCS_SEARCH_RESULTS) $(PUBLIC_DOCS_ANSWERED_QUESTIONS)

evaluate_code: moulinette
	./moulinette_pkg/moulinette-ubuntu evaluate_student_search_results $(PUBLIC_CODE_SEARCH_RESULTS) $(PUBLIC_CODE_ANSWERED_QUESTIONS)

.PHONY: run install debug clean lint lint-strict data moulinette index search search_dataset answer answer_dataset evaluate recall
