"""
File che funge da "wrapper" per LangChain. Unisce la logica del KG al formato
di tool atteso da LangChain. 
"""

from langchain_core.tools import tool
from prompts.tool_prompts import (
    KG_QUERY_PROMPT,
    KG_UPDATE_PROMPT,
    KG_TOPICS_OVERVIEW_PROMPT,
    KG_CONTEXT_PROMPT,
)
from knowledge_graph.queries import kg_topic_history, kg_topics_overview, kg_topic_context
from knowledge_graph.updater import update_kg_data


@tool(description=KG_QUERY_PROMPT)
def query_knowledge_graph(topic: str) -> str:
    """Interroga il Knowledge Graph Neo4j per la cronologia dei post su un argomento."""
    return kg_topic_history(topic)


@tool(description=KG_TOPICS_OVERVIEW_PROMPT)
def list_blog_topics() -> str:
    """Elenca tutti i topic del blog ordinati dal piu' trascurato (per trovare i gap)."""
    return kg_topics_overview()


@tool(description=KG_CONTEXT_PROMPT)
def get_editorial_context(topic: str) -> str:
    """Recupera post, claim, fonti e topic correlati per garantire coerenza in stesura."""
    return kg_topic_context(topic)


@tool(description=KG_UPDATE_PROMPT)
def update_knowledge_graph(
    topic: str,
    post_title: str,
    category: str,
    sources: list[str],
    claims: list[str],
    related_topics: list[str] = None,
    content: str = "",
    seo_score: float = None,
    cover_image: str = "",
) -> str:
    """Aggiorna (incrementalmente) il Knowledge Graph con un articolo approvato."""
    return update_kg_data(
        topic=topic,
        post_title=post_title,
        category=category,
        sources=sources,
        claims=claims,
        related_topics=related_topics,
        content=content,
        seo_score=seo_score,
        cover_image=cover_image,
    )
