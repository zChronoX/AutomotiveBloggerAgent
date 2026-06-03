"""
Server MCP di ricerca web (transport HTTP streamable).

Si avvia UNA VOLTA come servizio persistente (in un terminale dedicato) e resta
in ascolto su http://127.0.0.1:8765/mcp. L'agente lo chiama via HTTP a ogni ricerca.

Uso (dalla radice del progetto):
    python -m mcp_server.search_server
"""

import os
import sys

# Questo script vive nella sottocartella mcp_server/, ma importa moduli che stanno nella
# RADICE del progetto (config, prompts, ecc.). Aggiungiamo la cartella genitore
# al path di ricerca dei moduli.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Silenziamo i warning prima degli import che li generano (es. deprecation di Tavily),
# per non sporcare i log del server.
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import json
from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from prompts.tool_prompts import MCP_SUMMARIZER_PROMPT
from config.settings import Configuration

load_dotenv()
cfg = Configuration()

# ============================================================
# SERVER MCP via HTTP (transport "streamable-http")
# ============================================================
# Servizio persistente: si avvia UNA VOLTA in un terminale dedicato
#   python -m mcp_server.search_server
# e resta in ascolto. Il client (tools/mcp_client_tool.py) lo chiama via HTTP.
# Su Windows questo e' piu' affidabile dello stdio (che ha problemi noti con la
# gestione dei sottoprocessi/stream).
mcp_server = FastMCP("AutomotiveSearchServer", host="127.0.0.1", port=8765)

# Tavily: 1 ricerca (vedi MAX_WEB_SEARCHES=1 nel grafo) con 5 risultati ma SENZA contenuto
# grezzo. Test empirico: con include_raw_content=True la sintesi del riassuntore locale
# arrivava a ~2m30s (totale ~5min), troppo per un guadagno di qualita' marginale (il modello
# scrittore sintetizza comunque). Senza raw_content i tempi calano nettamente e la bozza
# resta di ottima qualita'. 5 risultati danno materiale sufficiente.
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_raw_content=False,
)

# Modello RIASSUNTORE: con HTTP il server e' persistente, quindi NON usiamo keep_alive=0:
# il modello resta caricato tra una ricerca e l'altra (le ricerche successive sono veloci).
summarizer_llm = ChatOllama(
    model=cfg.summarizer_model_name,
    num_ctx=cfg.summarizer_num_ctx,
    temperature=cfg.summarizer_temperature,
)


def _extract_text_from_tavily(raw_results) -> str:
    """Estrae il testo dai risultati Tavily in modo robusto ai vari formati possibili."""
    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except Exception:
            return raw_results.strip()

    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", [raw_results])

    if not isinstance(raw_results, list):
        return str(raw_results).strip()

    pieces = []
    for item in raw_results:
        if isinstance(item, dict):
            text = item.get("content") or item.get("raw_content") or item.get("snippet") or ""
            if text:
                pieces.append(text)
        elif isinstance(item, str):
            pieces.append(item)
    return "\n\n".join(pieces).strip()


def _extract_sources_from_tavily(raw_results) -> list[dict]:
    """
    Estrae gli URL (e i titoli) delle fonti dai risultati Tavily.

    Questi URL sono le PAGINE REALI da cui proviene il riassunto: vanno restituiti
    insieme al riassunto, cosi' il modello scrittore puo' citare la pagina precisa
    nel post (es. "https://www.motor1.com/...") invece di scrivere "[mcp_web_search]".
    """
    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except Exception:
            return []

    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", [raw_results])

    if not isinstance(raw_results, list):
        return []

    sources = []
    seen_urls = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            title = item.get("title") or url
            sources.append({"title": title, "url": url})
    return sources


@mcp_server.tool()
def search_and_summarize(query: str) -> str:
    """
    Esegue una ricerca web estesa su argomenti automotive (Tavily) e usa il modello
    riassuntore locale per produrre un riassunto tecnico denso del materiale trovato.

    Restituisce il riassunto SEGUITO dalla lista delle FONTI (URL reali) da cui
    proviene, cosi' il modello scrittore puo' citare le pagine precise nel post.
    """
    try:
        raw_results = tavily_search.invoke({"query": query})
        full_text = _extract_text_from_tavily(raw_results)
        sources = _extract_sources_from_tavily(raw_results)

        if not full_text.strip():
            return "Nessuna informazione trovata sul web per questa query."

        prompt = MCP_SUMMARIZER_PROMPT.format(full_text=full_text)
        response = summarizer_llm.invoke([HumanMessage(content=prompt)])
        summary = response.content

        # Accodiamo le fonti reali (URL) al riassunto. Il modello scrittore le citera'
        # nel post al posto del nome del tool. Formato chiaro e parsabile.
        if sources:
            sources_block = "\n".join(
                f"- {s['title']}: {s['url']}" for s in sources
            )
            return (
                f"{summary}\n\n"
                f"=== FONTI WEB (cita questi URL nel post, non il nome del tool) ===\n"
                f"{sources_block}"
            )
        return summary

    except Exception as e:
        return f"Errore nel Server MCP durante la ricerca o la sintesi: {str(e)}"


if __name__ == "__main__":
    # Avvio del server come servizio HTTP persistente (terminale dedicato).
    print("[MCP SERVER] Avvio su http://127.0.0.1:8765/mcp ...", file=sys.stderr)
    print("[MCP SERVER] Lascia questa finestra aperta. Premi CTRL+C per fermare.", file=sys.stderr)
    mcp_server.run(transport="streamable-http")
