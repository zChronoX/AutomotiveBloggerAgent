"""
Tool analogo al KG, serve come wrapper per il database vettoriale locale.

"""

from langchain_core.tools import tool
from prompts.tool_prompts import RAG_RETRIEVAL_PROMPT
from rag.retriever import retrieve_local


@tool(description=RAG_RETRIEVAL_PROMPT)
def retrieve_local_documents(query: str) -> str:
    """Recupera dai documenti locali (ChromaDB) i frammenti piu' rilevanti per la query."""
    return retrieve_local(query)
