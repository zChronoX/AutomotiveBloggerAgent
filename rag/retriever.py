"""
Modulo che cerca nei documenti locali (quindi in ChromaDB) e
torna i chunk più rilevanti con due soglie diverse
I metodi sono tracciabili con LangSmith (compaiono nella waterfall)
"""

import os
from config.settings import Configuration
from .vectorstore import vectorstore, PERSIST_DIR

# @traceable rende la funzione visibile nella waterfall di LangSmith.
try:
    from langsmith import traceable
except Exception:
    def traceable(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _wrap

_cfg = Configuration()

# Quanti chunk da recuperare in totale.
TOP_K = _cfg.rag_top_k

# Sogliia iniziale, tutto ciò che sta sotto è pertinente, sopra invece no.
DISTANCE_THRESHOLD = _cfg.rag_distance_threshold

# Seconda soglia usata, nel caso in cui la prima dovesse scartare proprio tutto.
# Oltre questa soglia non consideriamo i chunk. Meglio non prendere nulla che prendere qualcosa di sbagliato.
FALLBACK_MAX = _cfg.rag_fallback_max



# Metodo che estrae il nome del file dal chunk. 
def _doc_source(doc) -> str:
    """Estrae il nome del file di origine dai metadati del chunk (se presente)."""
    meta = getattr(doc, "metadata", {}) or {}
    src = meta.get("source") or meta.get("file_path") or meta.get("filename") or ""
    if src:
        return os.path.basename(src)
    return "documento locale"


# Funzione che recupera i documenti dal database locale (ChromaDB).
# Attraverso una query, vediamo se ci sono documenti rilevanti (con similarità)
# Ordiniamo in base alla distanza crescente
# Torniamo i chunk pertinenti. Includiamo anche il nome del file per correttezza.
@traceable(run_type="retriever", name="rag_local_retrieval")
def retrieve_local(query: str) -> str:
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

        # Diagnostica di debug non visibile di default (guardare l'env.)
        if _cfg.debug:
            print("[DIAG RAG] distanze:", [round(s, 3) for _, s in scored])

        # Filtro per distanza (soglia di pertinenza)
        filtered = [(doc, s) for doc, s in scored if s <= DISTANCE_THRESHOLD]

        # Se la prima soglia non fa prendere nessun chunk
        # usiamo una soglia più flessibile, ma sotto un valore massimo, 
        # prendiamo il chunk più rilevante. Altrimenti non prendiamo nulla.
        if not filtered:
            best_doc, best_dist = scored[0]
            if best_dist <= FALLBACK_MAX:
                filtered = [(best_doc, best_dist)]
            else:
                if _cfg.debug:
                    print(f"[DIAG RAG] miglior distanza {best_dist:.3f} > fallback {FALLBACK_MAX}: nessuna fonte.")
                return "Nessun documento locale pertinente trovato per questo tema."

        # Includiamo il nome del file e la distanza per la tracciabilita'
        parts = []
        for i, (doc, dist) in enumerate(filtered):
            src = _doc_source(doc)
            parts.append(f"[Fonte {i+1} - {src} (distanza {round(dist, 3)})]: {doc.page_content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Errore durante il recupero RAG: {str(e)}"
