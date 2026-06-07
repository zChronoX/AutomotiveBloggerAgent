"""
Modulo di configurazione del RAG, dove definisco
tutti i parametri e inizializzo gli oggetti
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Silenzio i warning innocui come :"embeddings.position_ids UNEXPECTED"
import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*position_ids.*")

# Parametri condivisi
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
