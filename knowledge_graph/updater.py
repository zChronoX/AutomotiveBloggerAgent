"""
Aggiornamento incrementale del Knowledge Graph.
Il KG viene aggiornato SOLO dopo approvazione dell'utente (requisito HITL).

Codice estratto da kg_tool.py (funzione update_knowledge_graph).
"""

from typing import List, Optional
from .client import get_db_driver, open_session


def update_kg_data(
    topic: str,
    post_title: str,
    category: str,
    sources: List[str],
    claims: List[str],
    related_topics: Optional[List[str]] = None,
    content: str = "",
    seo_score: Optional[float] = None,
    cover_image: str = "",
) -> str:
    """Aggiorna (incrementalmente) il Knowledge Graph con un articolo approvato.

    Il topic in ingresso e' gia' una chiave canonica (vedi canonical_topic). Prima di
    salvare, lo risolviamo SEMANTICAMENTE verso un topic esistente: se nel KG c'e' gia'
    un soggetto equivalente (es. "giulia quadrifoglio" vs "alfa romeo giulia quadrifoglio"),
    agganciamo QUELLO invece di creare un nodo quasi-duplicato. Cosi' il grafo non si
    frammenta e la gap-analysis resta affidabile.
    """
    # Risoluzione semantica del topic verso uno gia' esistente (se abbastanza simile).
    try:
        from .queries import resolve_topic
        topic = resolve_topic(topic)
    except Exception:
        # Se la risoluzione fallisce, usiamo il topic canonico cosi' com'e'.
        topic = (topic or "").lower().strip()

    query = """
    // 1. Crea/Trova Topic e Post (con timestamp alla creazione)
    MERGE (t:Topic {name: $topic})
    MERGE (p:Post {title: toLower($post_title)})
    ON CREATE SET p.category = $category, p.created_at = datetime(), p.content = $content,
                  p.seo_score = $seo_score, p.cover_image = $cover_image
    ON MATCH  SET p.category = $category, p.content = $content,
                  p.seo_score = $seo_score, p.cover_image = $cover_image
    MERGE (p)-[:COVERS_TOPIC]->(t)

    // 2. Topic correlati (relationships between topics)
    FOREACH (rel_topic IN $related_topics |
        MERGE (rt:Topic {name: toLower(rel_topic)})
        MERGE (t)-[:RELATED_TO]->(rt)
    )

    // 3. Fonti (sources used)
    FOREACH (url IN $sources |
        MERGE (s:Source {url: url})
        MERGE (p)-[:BASED_ON]->(s)
    )

    // 4. Key claims (key claims extracted)
    FOREACH (claim_text IN $claims |
        MERGE (c:Claim {text: claim_text})
        MERGE (p)-[:ASSERTS]->(c)
    )
    """
    try:
        driver = get_db_driver()
        with open_session(driver) as session:
            session.run(
                query,
                topic=topic,
                post_title=post_title,
                category=category,
                sources=sources,
                claims=claims,
                related_topics=related_topics or [],
                content=content,
                seo_score=seo_score,
                cover_image=cover_image,
            )
        return f"Knowledge Graph aggiornato per l'articolo '{post_title}' con fonti, claim e relazioni."
    except Exception as e:
        return f"Errore durante l'aggiornamento di Neo4j: {str(e)}"
    finally:
        if "driver" in locals():
            driver.close()
