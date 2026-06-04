"""
Tool di ricerca web tramite MCP, basato su langchain-mcp-adapters.

Usa la libreria ufficiale del notebook 3 del tutorial Deep Research
(langchain_mcp_adapters.MultiServerMCPClient) ma con transport HTTP "streamable_http"
invece di stdio: su Windows lo stdio ha problemi noti con la gestione dei sottoprocessi,
mentre l'HTTP con server persistente e' affidabile e gia' testato nel progetto.

ARCHITETTURA: il server MCP (mcp_server/search_server.py) va avviato a parte come
servizio persistente, in un terminale dedicato:
    python -m mcp_server.search_server
Il client si collega via HTTP a ogni ricerca.

Il nostro grafo e' sincrono, quindi incapsuliamo la chiamata MCP (async) in un wrapper
@tool sincrono eseguito in un event loop dedicato.
"""

import asyncio
import logging
import concurrent.futures

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from prompts.tool_prompts import WEB_SEARCH_PROMPT

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("mcp").setLevel(logging.CRITICAL)

MCP_SERVER_URL = "http://127.0.0.1:8765/mcp"
MCP_TIMEOUT_SECONDS = 90

# Configurazione MCP in stile notebook 3, con transport HTTP (streamable_http).
MCP_CONFIG = {
    "automotive_search": {
        "url": MCP_SERVER_URL,
        "transport": "streamable_http",
    }
}

# Client MCP inizializzato in modo lazy (come get_mcp_client nel notebook 3).
_client = None


def _get_mcp_client() -> MultiServerMCPClient:
    """Restituisce (creandolo una sola volta) il client MCP multi-server."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(MCP_CONFIG)
    return _client


def _extract_text(result) -> str:
    """
    Estrae il testo pulito dalla risposta del tool MCP. langchain-mcp-adapters puo'
    restituire: una stringa gia' pronta, oppure una lista di blocchi tipo
    [{'type': 'text', 'text': '...'}], oppure una tupla (content, artifact). Qui
    normalizziamo tutti i casi a testo semplice, evitando che tag e '\\n' letterali
    finiscano nell'output (com'era con un str() grezzo sulla struttura).
    """
    if result is None:
        return ""
    if isinstance(result, tuple) and result:
        result = result[0]
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or block.get("content", "") or "")
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(p for p in parts if p)
    if hasattr(result, "text"):
        return result.text or ""
    if hasattr(result, "content"):
        c = result.content
        return c if isinstance(c, str) else _extract_text(c)
    return str(result)


async def _run_mcp_search(query: str) -> str:
    """Ottiene i tool dal server MCP e invoca 'search_and_summarize' (async)."""
    client = _get_mcp_client()
    mcp_tools = await client.get_tools()
    search_tool = next((t for t in mcp_tools if t.name == "search_and_summarize"), None)
    if search_tool is None:
        return "Il server MCP non espone il tool 'search_and_summarize'."
    result = await search_tool.ainvoke({"query": query})
    text = _extract_text(result).strip()
    return text if text else "Il server MCP non ha restituito alcun testo."


def _run_in_dedicated_loop(query: str) -> str:
    """Esegue la coroutine MCP in un event loop nuovo e dedicato (contesto sync del grafo)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            asyncio.wait_for(_run_mcp_search(query), timeout=MCP_TIMEOUT_SECONDS)
        )
    finally:
        loop.close()


@tool(description=WEB_SEARCH_PROMPT)
def mcp_web_search(query: str) -> str:
    """Tool di ricerca web: delega ricerca+sintesi al server MCP HTTP (via langchain-mcp-adapters)."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_dedicated_loop, query)
            return future.result(timeout=MCP_TIMEOUT_SECONDS + 10)
    except concurrent.futures.TimeoutError:
        return (f"La ricerca web non ha risposto entro {MCP_TIMEOUT_SECONDS}s. "
                "Verifica che il server MCP sia avviato. Procedo senza fonti web.")
    except Exception as e:
        return (f"Errore di comunicazione col Server MCP ({str(e)}). "
                "Assicurati di aver avviato 'python -m mcp_server.search_server' "
                "in un terminale separato. Procedo senza fonti web.")
