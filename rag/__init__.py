"""
Modulo RAG (Retrieval-Augmented Generation).

Requisiti soddisfatti (dal PDF delle specifiche):
- Document retrieval da fonti esterne o locali (RAG).
- Integrazione delle informazioni del KG nel processo di retrieval (K-RAG via agent/nodes.py).
- Uso di conoscenza strutturata (KG) e documenti non strutturati (RAG).
"""

from .retriever import retrieve_local
from .vectorstore import vectorstore, PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

__all__ = [
    "retrieve_local",
    "vectorstore",
    "PERSIST_DIR",
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
]
