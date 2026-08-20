*This project has been created as part of the 42 curriculum by flauweri.*

# RAG against the machine

Will you answer my questions?

## Description

This project builds a local Retrieval-Augmented Generation (RAG) pipeline over the vLLM repository. The goal is to ingest a large codebase and documentation tree, segment it into searchable chunks, retrieve the most relevant passages for a user question, and answer that question with citations to the original source files while staying grounded in the retrieved context.

The project centers on a lexical retrieval system with a BM25 index, an LLM answer step, and an evaluation layer computing recall@k against a public dataset of reference questions.

The system is implemented as a Python CLI and is designed to work locally without depending on a remote API for the retrieval and answer pipeline.

## Instructions

### Requirements

- Python 3.13+
- `uv`
- Ollama installed and running locally
- Download vllm-0.10.1.zip

### Installation

From another terminal:

`ollama serve`

From the project root:

`make install`

### Common commands

```bash
make install
make run
make debug
make clean
make lint

make index
uv run -m src index [--dir <dir path>] [--max_chunk_size=<n>] [--embedding=<bool>] [--incremental=<bool>]
make search
uv run -m src search --query "<query>" [--k=<n>] [--hybrid=<bool>] [--expansion=<bool>]
make search_dataset
uv run -m src search_dataset <dataset_path> [--k=<n>] [--save_directory <dir path>] [--hybrid=<bool>] [--expansion=<bool>]
make answer
uv run -m src answer --query "<query>" [--k=<n>] [--caching=<bool>] [--hybrid=<bool>] [--expansion=<bool>]
make answer_dataset
uv run -m src answer_dataset <student_search_results_path> [--save_directory <dir path>] [--caching=<bool>]
make evaluate_docs & make evaluate_code
uv run -m src evaluate <student_search_results_path> <dataset_path>
```

## System architecture

The architecture follows the standard RAG pattern, adapted to a code/documentation corpus:

1. Ingestion layer
   - The project scans the raw repository tree under `data/raw/`.
   - Files are opened and parsed according to their extension (`.py`, `.md`, `.txt`).

2. Chunking layer
   - Each file is split into chunks using `RecursiveCharacterTextSplitter` from LangChain.
   - Each chunk keeps metadata including the source file path and character offsets, which is essential for both retrieval quality and evaluation.

3. Indexing layer
   - A BM25 lexical index is built with `bm25s`.
   - Persisted data is stored in `data/processed/` to avoid rebuilding everything on every run.

4. Retrieval layer
   - A `Search` component performs BM25 lookup.
   - Optional query expansion can be used before retrieval to enrich the initial user question.

5. Answer generation layer
   - The `Answer` component loads the top retrieved sources.
   - It reconstructs the relevant context from the original files.
   - It prompts the local LLM to answer using only the retrieved sources, reducing hallucination risk.

6. Evaluation layer
   - The `Evaluator` compares retrieved source spans with the expected answer source using recall@k.
   - The evaluation checks overlap on the file and character range, using an IoU-like criterion.

This is implemented mainly in:

- `src/index.py` for chunking and indexing;
- `src/search.py` for BM25 and query expansion;
- `src/answer.py` for answer generation from retrieved context;
- `src/evaluate.py` for retrieval-quality assessment;
- `src/rag.py` as the public CLI orchestrator;
- `src/config.py`, `src/models.py`, and supporting modules.

## Chunking strategy

The chunking strategy is central to the quality of the retrieval pipeline.

- Default chunk size: `2000` characters
- Chunk overlap: `20%` of the chunk size, which preserves continuity across section boundaries
- File-aware splitting:
  - Python files use `Language.PYTHON`
  - Markdown files use `Language.MARKDOWN`
  - Plain text falls back to a generic recursive text splitter
- Metadata preservation:
  - `file_path`
  - `first_character_index`
  - `last_character_index`

This design provides a good trade-off between context richness and retrieval precision. Smaller chunks are easier to match, while larger chunks keep enough surrounding context to support reasoning on technical code and documentation sections.

## Retrieval method

### Lexical retrieval

BM25 is used to score document chunks based on term frequency and document distribution, with a length-normalization effect. This gives strong results on code identifiers, API names, configuration keys, and technical vocabulary.

The retrieval process can optionally use query expansion before retrieval. This is done through a DSPy-based signature that rewrites the user question into more retrieval-friendly keywords.

## Bonus

These optional/advanced features are implemented or can be enabled as extras:

- **Semantic embeddings:** add a vector index built with a lightweight CPU embedding model (for example `all-MiniLM-L6-v2`) alongside the BM25 lexical index; store vectors in ChromaDB and use them for dense semantic matching.
- **Hybrid retrieval:** combine lexical (BM25) and semantic (embedding) rankings into a single fused result list (for example via Reciprocal Rank Fusion) to leverage both exact keyword matches and semantic similarity.
- **Incremental indexing:** track files via a manifest and re-index only changed files instead of rebuilding the entire index, reducing indexing time for large codebases.
- **Caching:** cache indexes and query results to speed up cold starts and repeated queries (local on-disk caches for BM25 metadata and serialized query responses).
- **Local HTTP API:** expose querying and answering over a small FastAPI-based HTTP API so the system can be driven by tools other than the CLI.
- **Query expansion:** (optional) use a small LM-based rewriter (DSPy or similar) to expand or rewrite user queries into more retrieval-friendly variants before performing lexical/semantic lookup.

## Performance analysis

The project includes an evaluator that measures recall@k against a set of public answered questions.

The evaluation pipeline:

- loads a dataset of question-answer pairs;
- compares the retrieved source ranges to the reference source range;
- computes whether the correct file and relevant span appear within the top `k` results;
- reports recall for `k = 1, 3, 5, 10`.

The evaluation logic is implemented in `src/evaluate.py`, and the project provides dedicated commands:

- `make recall_docs`
- `make recall_code`
- `make evaluate_docs`
- `make evaluate_code`

## Design decisions

Several implementation choices were guided by robustness and maintainability:

- Persistent storage for BM25 indexes under `data/processed/`
- Pydantic models for clear typed data structures and JSON serialization
- Local execution with Ollama to keep the pipeline runnable on a workstation or dev environment
- CLI-first design with `Fire`, making the project easy to run from the terminal

## Challenges faced

The main technical challenges were the following:

1. Large repository size and indexing cost
   - The dataset contains a large code tree, so the project had to avoid unnecessary re-indexing.

2. Chunk quality versus retrieval quality
   - Very large chunks reduce splitting precision, while very small chunks may fragment context.
   - The project uses a balanced chunk size and overlap to preserve source continuity.

3. Retrieval mismatch between code and documentation
   - Code contains identifiers, symbols, and patterns that differ from natural-language wording.

4. Grounding the answer in the correct source
   - LLMs are prone to hallucination if the context is insufficient or too noisy.
   - The answer layer explicitly reconstructs context from the retrieved file slices and instructs the model to answer only from that data.

5. Local model setup and dependencies
   - Ollama models must be downloaded locally before indexing and answering.
   - The project includes installation targets that pull the required models and sets clear error messages when collections or indexes are missing.

## Performance analysis examples

```bash
make search_dataset
make answer_dataset
```

## Resources

### Technical references

- Fire documentation: https://python-fire.readthedocs.io/en/latest/
- LangChain text splitters: https://reference.langchain.com/python/langchain-text-splitters
- Pydantic docs: https://pydantic.dev/docs/validation/latest/get-started
- Pathlib documentation: https://docs.python.org/3/library/pathlib.html
- BM25S documentation: https://bm25s.github.io/
- tqdm documentation: https://tqdm.github.io/
- Ollama model library: https://ollama.com/library

### AI usage in this project

1. Query expansion
   - A DSPy-based LM rewrites the original question into more retrieval-friendly terms, improving lexical recall for ambiguous queries.

2. Answer generation
   - The final answer is generated by a local Qwen model using the retrieved source snippets as context.
   - The model is instructed to answer only from the provided evidence to minimize hallucinations.

## Project structure

```text
.
├── Makefile
├── README.md
├── pyproject.toml
├── src/
│   ├── __main__.py
│   ├── __init__.py
│   ├── index.py
│   ├── search.py
│   ├── answer.py
│   ├── evaluate.py
│   ├── rag.py
│   ├── config.py
│   ├── errors.py
│   ├── models.py
│   └── ...
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
├── zip/
└── assets/
```

## Summary

This project implements a compact RAG system for a large code and documentation corpus focusing on lexical retrieval, chunking, generation, and evaluation. It is intended as both a practical engineering project and a local benchmark for understanding how retrieval quality influences final answer quality.
