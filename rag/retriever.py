"""
Logica di retrieval dal vector store locale (ChromaDB).
Funzione core (Python puro), richiamabile dal tool wrapper o direttamente.

Tracciata con @traceable di LangSmith: la chiamata appare nella waterfall come
step "retriever" dedicato, con la query in input e i documenti recuperati in output
(o il messaggio di "nessun documento" se non trova nulla).

Codice estratto da rag_tool.py.
"""

import os
from config.settings import Configuration
from .vectorstore import vectorstore, PERSIST_DIR

# @traceable rende la funzione visibile nella waterfall di LangSmith. Import difensivo:
# se langsmith non e' installato o il tracing e' spento, usiamo un decoratore no-op.
try:
    from langsmith import traceable
except Exception:
    def traceable(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        # Supporta sia @traceable sia @traceable(...)
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _wrap

_cfg = Configuration()

# Recuperiamo piu' candidati (k) e poi filtriamo per distanza, cosi' all'agente
# arrivano solo i chunk davvero pertinenti, ordinati dal migliore.
TOP_K = 5

# Soglia di distanza PERMISSIVA, tarata empiricamente sui documenti reali
# (vedi diagnostics/tune_rag_threshold.py): i chunk pertinenti per tema osservati
# stavano fino a ~1.15 (alcuni temi "difficili" come il tagliando ibride fino a ~1.145).
# 1.175 li cattura tutti lasciando il filtro fine di rilevanza al Self-RAG.
DISTANCE_THRESHOLD = 1.175


def _doc_source(doc) -> str:
    """Estrae il nome del file di origine dai metadati del chunk (se presente)."""
    meta = getattr(doc, "metadata", {}) or {}
    src = meta.get("source") or meta.get("file_path") or meta.get("filename") or ""
    if src:
        return os.path.basename(src)
    return "documento locale"


@traceable(run_type="retriever", name="rag_local_retrieval")
def retrieve_local(query: str) -> str:
    """Recupera dai documenti locali (ChromaDB) i frammenti piu' rilevanti per la query.

    L'output include il NOME DEL FILE di origine di ogni chunk, cosi' nella waterfall
    di LangSmith si vede esattamente da quali documenti locali ha attinto l'agente.
    """
    if not os.path.isdir(PERSIST_DIR):
        return ("Il database locale non esiste ancora. Esegui prima 'python -m rag.ingest' "
                "per indicizzare i documenti.")
    try:
        # Recupero con punteggio di distanza (piu' basso = piu' simile)
        scored = vectorstore.similarity_search_with_score(query, k=TOP_K)
        if not scored:
            return "Nessun documento rilevante trovato nel database locale."

        # Ordiniamo per distanza crescente (i piu' pertinenti per primi)
        scored.sort(key=lambda x: x[1])

        # DIAGNOSTICA: visibile solo con DEBUG=true nel .env
        if _cfg.debug:
            print("[DIAG RAG] distanze:", [round(s, 3) for _, s in scored])

        # Filtro permissivo per distanza
        filtered = [(doc, s) for doc, s in scored if s <= DISTANCE_THRESHOLD]

        # Fallback: se il filtro azzera tutto, teniamo comunque il migliore
        # (meglio una fonte debole che nessuna fonte).
        if not filtered:
            filtered = [scored[0]]

        # Includiamo il nome del file e la distanza per la tracciabilita'
        parts = []
        for i, (doc, dist) in enumerate(filtered):
            src = _doc_source(doc)
            parts.append(f"[Fonte {i+1} - {src} (distanza {round(dist, 3)})]: {doc.page_content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Errore durante il recupero RAG: {str(e)}"
