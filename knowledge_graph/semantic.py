"""
Script che mi aiuta con la semantica nel KG. Nel caso in cui confrontassi stringhe di topic
leggermente diverse, il KG li vedrebbe come due argomenti completamente differenti e potrei
finire per scrivere più volte articoli sugli stessi argomenti andando contro il principio di unicità del KG.
Ho deciso di usare gli embedding del modello all-MiniLM-L6-v2 per confrontare i topic del KG.
Invece che confrontare stringe, trasformo i topic in vettori numerici come il RAG, confronto le
similarità coseno tra vettori, se supero una certa soglia (che ho testato con un altro script)
allora è lo stesso topic.
"""

import numpy as np
from rag.vectorstore import embeddings  
import os


# Posso impostare la similarità dall'env, in alternativa ne metto una già testata.
try:
    SIMILARITY_THRESHOLD = float(os.getenv("KG_TOPIC_SIM_THRESHOLD", "0.75"))
except ValueError:
    SIMILARITY_THRESHOLD = 0.75


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_semantic_match(query_topic: str, candidate_topics: list[str], threshold: float = None):
    """
    Applico quanto scritto sopra.
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    if not query_topic or not candidate_topics:
        return None, 0.0

    try:
        q_emb = embeddings.embed_query(query_topic)
        cand_embs = embeddings.embed_documents(candidate_topics)
    except Exception:
        # Se l'embedding fallisce torno 0. Si può usare un alternativa di confronto tra stringhe.
        return None, 0.0

    best_topic, best_score = None, -1.0
    for cand, emb in zip(candidate_topics, cand_embs):
        s = _cosine(q_emb, emb)
        if s > best_score:
            best_topic, best_score = cand, s

    if best_score >= threshold:
        return best_topic, best_score
    return None, best_score
