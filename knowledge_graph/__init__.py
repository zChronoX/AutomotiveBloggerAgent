"""
Modulo Knowledge Graph (Neo4j).

Requisiti soddisfatti (dal PDF delle specifiche):
- Mantiene il grafo editoriale con post, topic, fonti, claim e relazioni.
- Interrogato in fase di Topic Suggestion (gap analysis, evitare ripetizioni).
- Interrogato in fase di Post Drafting (coerenza, cross-link).
- Aggiornato incrementalmente dopo ogni post approvato (HITL).
"""

from .queries import kg_topic_history, kg_topics_overview, kg_topic_context
from .updater import update_kg_data
from .client import get_db_driver

__all__ = [
    "kg_topic_history",
    "kg_topics_overview",
    "kg_topic_context",
    "update_kg_data",
    "get_db_driver",
]
