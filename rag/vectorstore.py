"""
Configurazione e inizializzazione del vector store (ChromaDB).

I TRE parametri (PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL) sono il contratto
condiviso tra ingestione (rag/ingest.py) e retrieval (rag/retriever.py).
Se ne modifichi uno, aggiorna entrambi (ora basta farlo qui).
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# PARAMETRI CONDIVISI tra ingest e retriever
# ============================================================
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "blog_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding model (inizializzato una volta, riusato da retriever e ingest)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Vector store pronto per il retrieval
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)
