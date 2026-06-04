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
# Valori ora configurabili da .env (RAG_TOP_K, RAG_DISTANCE_THRESHOLD) senza toccare il codice.
TOP_K = _cfg.rag_top_k

# Soglia di distanza: piu' BASSA = piu' selettiva. Tarata su dati reali (1.10): tiene i
# pertinenti (fino a ~1.065) e scarta i non pertinenti (da ~1.216). Il filtro fine resta al Self-RAG.
DISTANCE_THRESHOLD = _cfg.rag_distance_threshold

# Soglia del FALLBACK: quando il filtro principale scarta tutto, teniamo il documento migliore
# SOLO se sotto questa soglia piu' permissiva. Oltre, niente fonte (meglio nessuna che sbagliata).
FALLBACK_MAX = _cfg.rag_fallback_max


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

        # Filtro per distanza (soglia di pertinenza)
        filtered = [(doc, s) for doc, s in scored if s <= DISTANCE_THRESHOLD]

        # Fallback CONDIZIONATO: se il filtro azzera tutto, teniamo il documento migliore
        # SOLO se e' almeno entro FALLBACK_MAX. Se anche il migliore e' troppo lontano,
        # NON forziamo una fonte: e' meglio "nessun documento pertinente" che un documento
        # fuori tema (era la causa di fonti spurie, es. reti di bordo in un post su una moto).
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
