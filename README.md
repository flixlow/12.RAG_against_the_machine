# 12.RAG_against_the_machine

## Description

## Instructions

## Resources
- Fire doc: https://python-fire.readthedocs.io/en/latest/
- Langchain doc: https://reference.langchain.com/python/langchain-text-splitters
- Pydantic doc: https://pydantic.dev/docs/validation/latest/get-started
- Pathlib (rglob): https://docs.python.org/3/library/pathlib.html
- bm25s doc: https://bm25s.github.io/

- transformers
- dspy
- tqdm
- chromadb
- Qwen/Qwen3-0.6B
- asv
- dspy -> cache

## Brouillon
- ouvrir les fichiers selon si c'est des fichiers python ou de la doc
- chuncker
- indexer
- trier selon la pertinence
- Retrieval-Augmented Generation (RAG) system
1. Ingest the vLLM repository (provided as attachment) and create a searchable
knowledge base
2. Search this knowledge base to find relevant code snippets and documentation for
given questions
3. Answer questions using an LLM (Qwen/Qwen3-0.6B) with the retrieved context
4. Evaluate your retrieval system’s quality using recall@k metrics

[ ] tester avec chunk_size petit pour voir qu'il n'y est pas de probleme avec le chunk size overlap * 0.05
[ ] pas sur du \n\n pour le separateur