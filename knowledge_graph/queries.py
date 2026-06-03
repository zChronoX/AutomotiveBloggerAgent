"""
Query di lettura del Knowledge Graph.
Funzioni "core" (Python puro) richiamabili in modo DETERMINISTICO dal grafo,
senza dipendere dal fatto che il modello scelga il tool giusto.

Tracciate con @traceable: nella waterfall di LangSmith compaiono come step dedicati
con il topic in input e il contesto KG in output.

Codice estratto da kg_tool.py (sezione FUNZIONI CORE).
"""

from .client import run_read, fmt_date

# @traceable rende le query KG visibili nella waterfall. Import difensivo no-op
# se langsmith non e' disponibile.
try:
    from langsmith import traceable
except Exception:
    def traceable(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _wrap


@traceable(run_type="retriever", name="kg_topic_history")
def kg_topic_history(topic: str) -> str:
    """
    Cronologia dei post su un topic (+ topic correlati), dal piu' recente.
    Usata in: fase di pianificazione (evitare doppioni) e drafting (coerenza).
    """
    posts_q = """
    MATCH (t:Topic {name: toLower($topic)})<-[:COVERS_TOPIC]-(p:Post)
    RETURN p.title AS title, p.category AS category, p.created_at AS created_at
    ORDER BY p.created_at DESC
    """
    related_q = """
    MATCH (t:Topic {name: toLower($topic)})-[:RELATED_TO]-(rt:Topic)
    RETURN DISTINCT rt.name AS related
    """
    try:
        posts = run_read(posts_q, topic=topic)
        if not posts:
            return f"L'argomento '{topic}' non e' mai stato trattato nel blog (e' un possibile gap di copertura)."

        lines = [f"L'argomento '{topic}' e' gia' stato trattato nei seguenti post:"]
        for r in posts:
            lines.append(f"- {r['title']} (Categoria: {r['category']}, del {fmt_date(r['created_at'])})")

        related = [r["related"] for r in run_read(related_q, topic=topic) if r["related"]]
        if related:
            lines.append("Topic correlati gia' presenti nel grafo: " + ", ".join(related))
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura di Neo4j: {str(e)}"


@traceable(run_type="retriever", name="kg_topics_overview")
def kg_topics_overview() -> str:
    """
    Panoramica di TUTTI i topic con numero di post e data dell'ultimo post,
    ordinati dal piu' trascurato (i topic mai coperti compaiono per primi = gap).
    Usata in: fase di pianificazione per individuare lacune e argomenti vecchi.
    """
    query = """
    MATCH (t:Topic)
    OPTIONAL MATCH (t)<-[:COVERS_TOPIC]-(p:Post)
    RETURN t.name AS topic, count(p) AS post_count, max(p.created_at) AS last_post
    ORDER BY post_count ASC, last_post ASC
    """
    try:
        rows = run_read(query)
        if not rows:
            return "Il Knowledge Graph e' vuoto: nessun topic registrato. Qualsiasi argomento e' nuovo."

        lines = ["Panoramica della copertura editoriale (dal piu' trascurato):"]
        for r in rows:
            if r["post_count"] == 0:
                lines.append(f"- {r['topic']}: MAI trattato (gap di copertura).")
            else:
                lines.append(
                    f"- {r['topic']}: {r['post_count']} post, ultimo il {fmt_date(r['last_post'])}."
                )
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura di Neo4j: {str(e)}"


@traceable(run_type="retriever", name="kg_topic_context")
def kg_topic_context(topic: str) -> str:
    """
    Contesto editoriale completo di un topic per garantire COERENZA e CROSS-LINK
    in fase di stesura: post esistenti, claim chiave gia' affermati, fonti usate,
    topic correlati. E' anche il pezzo di KG che alimenta il K-RAG.
    """
    query = """
    MATCH (t:Topic {name: toLower($topic)})
    OPTIONAL MATCH (t)<-[:COVERS_TOPIC]-(p:Post)
    OPTIONAL MATCH (p)-[:ASSERTS]->(c:Claim)
    OPTIONAL MATCH (p)-[:BASED_ON]->(s:Source)
    OPTIONAL MATCH (t)-[:RELATED_TO]-(rt:Topic)
    RETURN collect(DISTINCT p.title) AS posts,
           collect(DISTINCT c.text)  AS claims,
           collect(DISTINCT s.url)   AS sources,
           collect(DISTINCT rt.name) AS related
    """
    try:
        rows = run_read(query, topic=topic)
        if not rows:
            return f"Nessun contesto nel KG per '{topic}': e' un argomento nuovo, nessun vincolo di coerenza."

        r = rows[0]
        posts = [x for x in r["posts"] if x]
        claims = [x for x in r["claims"] if x]
        sources = [x for x in r["sources"] if x]
        related = [x for x in r["related"] if x]

        if not any([posts, claims, sources, related]):
            return f"Nessun contesto nel KG per '{topic}': e' un argomento nuovo, nessun vincolo di coerenza."

        lines = [f"Contesto del KG per il topic '{topic}' (usalo per coerenza e link interni):"]
        if posts:
            lines.append("Post esistenti collegati: " + "; ".join(posts))
        if claims:
            lines.append("Claim gia' affermati (non contraddirli): " + "; ".join(claims))
        if sources:
            lines.append("Fonti gia' usate: " + "; ".join(sources))
        if related:
            lines.append("Topic correlati per cross-link: " + ", ".join(related))
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura di Neo4j: {str(e)}"


@traceable(run_type="retriever", name="kg_related_topics")
def kg_related_topics(topic: str) -> list[str]:
    """
    Restituisce SOLO la lista dei topic correlati a un dato topic nel KG.
    Usata per la QUERY EXPANSION del K-RAG: le entita'/topic correlati del KG
    vengono accodati alla query di retrieval per espanderla (requisito:
    "use the Knowledge Graph to expand or refine retrieval queries").
    """
    query = """
    MATCH (t:Topic {name: toLower($topic)})-[:RELATED_TO]-(rt:Topic)
    RETURN DISTINCT rt.name AS related
    """
    try:
        rows = run_read(query, topic=topic)
        return [r["related"] for r in rows if r["related"]]
    except Exception:
        return []
