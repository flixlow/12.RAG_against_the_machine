# 12.RAG_against_the_machine

## Description

## Instructions

## Resources
- Fire doc: https://python-fire.readthedocs.io/en/latest/
- Langchain doc: https://reference.langchain.com/python/langchain-text-splitters
- Pydantic doc: https://pydantic.dev/docs/validation/latest/get-started
- Pathlib (rglob): https://docs.python.org/3/library/pathlib.html
- bm25s doc: https://bm25s.github.io/
- tqdm doc: https://tqdm.github.io/

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
- comment fonctionne bm25s ? TD-IDF, Saturation du TF, Normalisation par la longueur des documents

Ce qu’il reprend de TF-IDF
TF (term frequency) : plus un mot apparaît dans un document, plus il compte
IDF (inverse document frequency) : un mot rare dans le corpus est plus important qu’un mot très fréquent
Ce que BM25 change par rapport à TF-IDF

BM25 ajoute deux idées importantes :

1) Saturation du TF

Dans TF-IDF, si un mot apparaît 50 fois, il devient 50× plus important (linéaire).

BM25 corrige ça :

au début, chaque occurrence aide beaucoup
puis ça “plafonne” (diminishing returns)

intuition : répéter 100 fois “chat” n’aide pas 100× plus.

2) Normalisation par la longueur des documents

Un long document a naturellement plus de mots.

BM25 corrige ça :

un mot dans un petit document “pèse” plus
un mot dans un long document est pénalisé


1. Ingest the vLLM repository (provided as attachment) and create a searchable
knowledge base
2. Search this knowledge base to find relevant code snippets and documentation for
given questions
3. Answer questions using an LLM (Qwen/Qwen3-0.6B) with the retrieved context
4. Evaluate your retrieval system’s quality using recall@k metrics

[ ] tester avec chunk_size petit pour voir qu'il n'y est pas de probleme avec le chunk size overlap * 0.05
[ ] pas sur du \n\n pour le separateur
[ ] peut etre refactore le code de index pour transformer en 4 classes, une pour la liste de fichier, une pour le open, une pour chunk et une pour l'indexage
[ ] choix de OSError pour les erreurs d'ouverture de fichiers


make install
make index
make search_dataset
make answer_dataset
make evaluate

[ ] search -> open file -> add to json file output in one time
[ ] revoir toute la gestion des erreurs
[ ] retirer les type ignore