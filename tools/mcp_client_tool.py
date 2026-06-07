"""
Tool di ricerca che chiama il server MCP. A differenza di stdio
che apparentemente da problemi, uso streamable_http. 
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

# Configurazione MCP con transport HTTP (streamable_http).
MCP_CONFIG = {
    "automotive_search": {
        "url": MCP_SERVER_URL,
        "transport": "streamable_http",
    }
}

_client = None

# Restituisce il client MCP multi-server
def _get_mcp_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        _client = MultiServerMCPClient(MCP_CONFIG)
    return _client

# Normalizzo il risultato ottenuto dal protocollo MCP, estraendo solo il testo pulito.
def _extract_text(result) -> str:
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

# Funzione asincrona che recupera il client, chiede al server i tool MCP che abbiamo (nel nostro caso 1)
# ottiene search_and_summarize, se non lo trova torna errore.
async def _run_mcp_search(query: str) -> str:
    client = _get_mcp_client()
    mcp_tools = await client.get_tools()
    search_tool = next((t for t in mcp_tools if t.name == "search_and_summarize"), None)
    if search_tool is None:
        return "Il server MCP non espone il tool 'search_and_summarize'."
    result = await search_tool.ainvoke({"query": query})
    text = _extract_text(result).strip()
    return text if text else "Il server MCP non ha restituito alcun testo."

# Creo un'evento isolato per la ricerca web. Ha un timeout di 90 secondi
# nel caso in cui ci sono stati errori con la ricerca (visto che solitamente impiega dai 40 ai 60 secondi).
def _run_in_dedicated_loop(query: str) -> str:
    """Esegue la coroutine MCP in un event loop nuovo e dedicato (contesto sync del grafo)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            asyncio.wait_for(_run_mcp_search(query), timeout=MCP_TIMEOUT_SECONDS)
        )
    finally:
        loop.close()

# Tool usato dall'agente per fare ricerche sul web.
# nel caso in cui il server MCP non risponda entro i tempi previsti
# o ci siano stati errori con la ricerca, avvisa l'agente e non fa bloccarsi l'intero processo.
@tool(description=WEB_SEARCH_PROMPT)
def mcp_web_search(query: str) -> str:
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
