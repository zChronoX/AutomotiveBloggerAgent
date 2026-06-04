"""
Configurazione e inizializzazione del vector store (ChromaDB).

I TRE parametri (PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL) sono il contratto
condiviso tra ingestione (rag/ingest.py) e retrieval (rag/retriever.py).
Se ne modifichi uno, aggiorna entrambi (ora basta farlo qui).
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Silenzia il warning innocuo "embeddings.position_ids UNEXPECTED" che compare al
# caricamento di all-MiniLM-L6-v2: e' dovuto a un campo obsoleto nel checkpoint del
# modello rispetto alla versione di transformers, NON indica un problema (il modello
# funziona correttamente). Abbassiamo il livello di logging per non sporcare l'output.
import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*position_ids.*")

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
