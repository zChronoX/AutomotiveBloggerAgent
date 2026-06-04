"""
Matching SEMANTICO dei topic del Knowledge Graph.

Invece di confrontare i nomi dei topic come stringhe esatte (fragile: "alfa romeo
giulia quadrifoglio" != "giulia quadrifoglio"), confrontiamo i loro EMBEDDING e
consideriamo "lo stesso topic" quelli con similarita' coseno sopra una soglia.

Riusa il modello di embedding gia' caricato per il RAG (all-MiniLM-L6-v2): nessun
modello aggiuntivo in VRAM.

Requisito di progetto soddisfatto: il KG riconosce i soggetti gia' trattati anche
se formulati in modo diverso, rendendo affidabili la gap-analysis (anti-ripetizione)
e la coerenza in drafting.
"""

import numpy as np
from rag.vectorstore import embeddings  # istanza HuggingFaceEmbeddings gia' inizializzata

# Soglia di similarita' coseno oltre la quale due topic sono considerati lo STESSO.
# VALORE DI PARTENZA: va verificato sul proprio hardware/dati. Con all-MiniLM-L6-v2,
# soggetti uguali formulati diversamente stanno tipicamente sopra ~0.75, soggetti
# diversi ben sotto. Se vedi falsi match, alza la soglia; se non aggancia i doppioni,
# abbassala. Configurabile via .env (KG_TOPIC_SIM_THRESHOLD).
import os
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
    Dato un topic cercato e la lista dei topic esistenti nel KG, restituisce il
    candidato semanticamente piu' vicino SE supera la soglia, altrimenti None.

    Ritorna: (topic_match, score) oppure (None, best_score) se sotto soglia.
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    if not query_topic or not candidate_topics:
        return None, 0.0

    try:
        q_emb = embeddings.embed_query(query_topic)
        cand_embs = embeddings.embed_documents(candidate_topics)
    except Exception:
        # Se l'embedding fallisce, nessun match semantico (il chiamante usera' il fallback)
        return None, 0.0

    best_topic, best_score = None, -1.0
    for cand, emb in zip(candidate_topics, cand_embs):
        s = _cosine(q_emb, emb)
        if s > best_score:
            best_topic, best_score = cand, s

    if best_score >= threshold:
        return best_topic, best_score
    return None, best_score
