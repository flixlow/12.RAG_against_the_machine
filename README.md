*This project has been created as part of the 42 curriculum by flixlow.*

# RAG against the machine

Will you answer my questions?

## Description

This project builds a local Retrieval-Augmented Generation (RAG) pipeline over the vLLM repository. The goal is to ingest a large codebase and documentation tree, segment it into searchable chunks, retrieve the most relevant passages for a user question, and answer that question with citations to the original source files while staying grounded in the retrieved context.

The project is designed around a hybrid retrieval system:

- a lexical index based on BM25, optimized for exact keyword matching and technical terms;
- a semantic index built with embeddings and stored in ChromaDB;
- an LLM answer step driven by Ollama and the local Qwen model;
- an evaluation layer computing recall@k against a public dataset of reference questions.

The system is implemented as a Python CLI and can also be launched through a small FastAPI interface. It was built to work locally, without depending on a remote API for the retrieval and answer pipeline.

## System architecture

The architecture follows the standard RAG pattern, but adapted to a code/documentation corpus:

1. Ingestion layer
   - The project scans the raw repository tree under `data/raw/`.
   - It detects new, modified, and deleted files using a manifest-based incremental index.
   - Files are opened and parsed according to their extension (`.py`, `.md`, `.txt`).

2. Chunking layer
   - Each file is split into chunks using `RecursiveCharacterTextSplitter` from LangChain.
   - Each chunk keeps metadata including the source file path and character offsets, which is essential for both retrieval quality and evaluation.

3. Indexing layer
   - A BM25 lexical index is built with `bm25s`.
   - A semantic vector index is created with ChromaDB and Ollama embeddings.
   - Persisted data is stored in `data/processed/` to avoid rebuilding everything on every run.

4. Retrieval layer
   - A `Search` component performs BM25 lookup, semantic lookup, and hybrid fusion.
   - Optional query expansion can be used before retrieval to enrich the initial user question.
   - Hybrid search combines both retrieval sources using Reciprocal Rank Fusion (RRF).

5. Answer generation layer
   - The `Answer` component loads the top retrieved sources.
   - It reconstructs the relevant context from the original files.
   - It prompts the local LLM to answer using only the retrieved sources, reducing hallucination risk.

6. Evaluation layer
   - The `Evaluator` compares retrieved source spans with the expected answer source using recall@k.
   - The evaluation checks overlap on the file and character range, using an IoU-like criterion.

This is implemented mainly in:

- `src/index.py` for chunking, indexing, and incremental update logic;
- `src/search.py` for BM25, vector search, hybrid ranking, and query expansion;
- `src/answer.py` for answer generation from retrieved context;
- `src/evaluate.py` for retrieval-quality assessment;
- `src/rag.py` as the public CLI orchestrator;
- `src/api.py` for local HTTP access.

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

The retrieval system is hybrid and combines lexical and semantic relevance.

### Lexical retrieval

BM25 is used to score document chunks based on term frequency and document distribution, with a length-normalization effect. This gives strong results on code identifiers, API names, configuration keys, and technical vocabulary.

### Semantic retrieval

A vector store is built with ChromaDB and an embedding model pulled through Ollama. Semantic retrieval is useful when the question uses paraphrases or related concepts not explicitly present in the query terms.

### Hybrid ranking

The hybrid search is implemented in `Search.reciprocal_rank_fusion()`:

- BM25 results and semantic results are both ranked;
- each result receives a score based on its reciprocal rank;
- the combined ranking is sorted by the fused score;
- the top `k` hits are returned.

This method is robust because it preserves the strengths of both retrieval methods without relying on a single ranking signal.

The retrieval process can also optionally use query expansion before retrieval. This is done through a DSPy-based signature that rewrites the user question into more retrieval-friendly keywords.

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

The repository notes target values of approximately 95% recall on documentation retrieval and around 70% on code retrieval, depending on the chosen chunk size, embedding model, and whether hybrid retrieval is enabled. In practice, these metrics are a good indicator of whether a retrieved answer is grounded in the right source region rather than only being fluent.

## Design decisions

Several implementation choices were guided by robustness and maintainability:

- Incremental indexing using a manifest file to avoid reprocessing unchanged files
- Persistent storage for BM25 and Chroma indexes under `data/processed/`
- Pydantic models for clear typed data structures and JSON serialization
- Local execution with Ollama to keep the pipeline fully runnable on a workstation or dev environment
- Query and response caching to avoid repeating expensive model calls on identical inputs
- CLI-first design with `Fire`, making the project easy to run from the terminal

## Challenges faced

The main technical challenges were the following:

1. Large repository size and indexing cost
   - The dataset contains a large code tree, so the project had to avoid unnecessary re-indexing.
   - The solution was incremental detection of changed files and manifest-based tracking.

2. Chunk quality versus retrieval quality
   - Very large chunks reduce splitting precision, while very small chunks may fragment context.
   - The project uses a balanced chunk size and overlap to preserve source continuity.

3. Retrieval mismatch between code and documentation
   - Code contains identifiers, symbols, and patterns that differ from natural-language wording.
   - Hybrid retrieval mitigates this by combining lexical and semantic signals.

4. Grounding the answer in the correct source
   - LLMs are prone to hallucination if the context is insufficient or too noisy.
   - The answer layer explicitly reconstructs context from the retrieved file slices and instructs the model to answer only from that data.

5. Local model setup and dependencies
   - Ollama models must be downloaded locally before indexing and answering.
   - The project includes installation targets that pull the required models and sets clear error messages when collections or indexes are missing.

## Instructions

### Requirements

- Python 3.13+
- `uv`
- Ollama installed and running locally
- Access to the repository data under `data/raw/`

### Installation

From the project root:

```bash
uv sync
ollama pull qwen3:0.6b
ollama pull all-minilm:l6-v2
```

The project also includes model pulls in the `Makefile` and a convenience `make install` target.

### Common commands

```bash
make install
make index
make search_dataset
make answer_dataset
make evaluate_docs
make evaluate_code
```

### Single query examples

```bash
uv run -m src search --query "How to configure OpenAI server?" --k 5
uv run -m src answer --query "How to configure OpenAI server?" --k 5
uv run -m src index --dir data/raw/vllm-0.10.1 --max_chunk_size 2000 --embedding
```

### Hybrid retrieval example

```bash
uv run -m src search_dataset data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json --k 5 --hybrid
```

### Local API

```bash
uv run -m fastapi dev src/api.py
```

Then the API can be used to query the index and ask questions through HTTP endpoints.

## Example usage

### Example 1: index the raw source tree

```bash
make index
```

This builds the BM25 index and creates chunk metadata under `data/processed/`.

### Example 2: search a single question

```bash
make search QUERY='How does the retrieval pipeline combine lexical and semantic scores?'
```

### Example 3: run retrieval over a question dataset

```bash
make search_dataset
```

### Example 4: answer the dataset questions using retrieved context

```bash
make answer_dataset
```

### Example 5: compute recall@k

```bash
make recall_docs
make recall_code
```

## Resources

### Technical references

- Fire documentation: https://python-fire.readthedocs.io/en/latest/
- LangChain text splitters: https://reference.langchain.com/python/langchain-text-splitters
- Pydantic docs: https://pydantic.dev/docs/validation/latest/get-started
- Pathlib documentation: https://docs.python.org/3/library/pathlib.html
- BM25S documentation: https://bm25s.github.io/
- tqdm documentation: https://tqdm.github.io/
- Chroma documentation: https://docs.trychroma.com/docs
- Ollama model library: https://ollama.com/library
- Embeddings overview: https://milvus.io/intro

### AI usage in this project

AI is used in three major parts of the pipeline:

1. Embeddings
   - Dense semantic retrieval is performed with embedding models served through Ollama.
   - These embeddings are stored in ChromaDB and compared with cosine similarity for semantic matching.

2. Query expansion
   - A DSPy-based LM rewrites the original question into more retrieval-friendly terms, improving lexical recall for ambiguous queries.

3. Answer generation
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
│   ├── api.py
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

This project implements a compact but full-stack RAG system for a large code and documentation corpus. It demonstrates the main steps of an industrial retrieval pipeline: ingestion, chunking, lexical and semantic indexing, hybrid retrieval, generation, and evaluation. It is intended as both a practical engineering project and a local benchmark for understanding how retrieval quality influences final answer quality.

http://127.0.0.1:8000/docs#/default/answer_answer_post