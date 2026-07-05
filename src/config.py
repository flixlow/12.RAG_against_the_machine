class Config:
    DATA = "data/"
    PROCESSED = DATA + "processed/"
    CHUNKS_PATH = PROCESSED + "chunks/chunks.json"
    CHROMA_PATH = PROCESSED + "chroma"
    BM25_PATH = PROCESSED + "bm25_index"
    DOCS_EM_MODEL = "nomic-embed-text"
    CODE_EM_MODEL = "nomic-embed-code"
