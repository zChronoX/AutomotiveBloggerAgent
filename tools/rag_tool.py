"""
Tool RAG esposto all'agente (ReAct).
Wrapper snello: delega la logica di retrieval a rag/retriever.py.
"""

from langchain_core.tools import tool
from prompts.tool_prompts import RAG_RETRIEVAL_PROMPT
from rag.retriever import retrieve_local


@tool(description=RAG_RETRIEVAL_PROMPT)
def retrieve_local_documents(query: str) -> str:
    """Recupera dai documenti locali (ChromaDB) i frammenti piu' rilevanti per la query."""
    return retrieve_local(query)
