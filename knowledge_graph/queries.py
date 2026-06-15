"""
Modulo di letture del Knowledge Graph senza modificarlo.
Funzioni chiamate in modo deterministico dal grafo,
senza dipendere dal fatto che il modello scelga il tool giusto.

Tracciate con @traceable così nella waterfall di LangSmith compaiono come step dedicati
con il topic in input e il contesto KG in output.

Tutti i metodi usano un try/except in modo da tornare stringhe/liste vuote
in caso ddi malfunzionamento, così il grafo non crasha.
"""

from .client import run_read, fmt_date
from .semantic import best_semantic_match

# @traceable rende le query KG visibili nella waterfall di LangSmith.
# try/except serve ad evitare errori se langsmith non è disponibile (o non installato).
try:
    from langsmith import traceable
except Exception:
    def traceable(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _wrap

# Query che prende tutti i topic del KG per poi fare il matching semantico
def _all_topic_names() -> list[str]:
    try:
        rows = run_read("MATCH (t:Topic) RETURN t.name AS name")
        return [r["name"] for r in rows if r.get("name")]
    except Exception:
        return []



# Applico la logica di risoluzione semantica del topic:
# lo normalizzo, prendo tutti i topic con la funzione sopra
# confronto tramite embeddings, se torna un valore sopra i 0.75
# allora lo considero lo stesso topic, altrimenti considero il nuovo topic.
def resolve_topic(topic: str) -> str:
    if not topic:
        return ""
    key = topic.lower().strip()
    candidates = _all_topic_names()
    match, score = best_semantic_match(key, candidates)
    return match if match else key



# Metodo usato dal planner per vedere quali sono
# stati i temi trattati, e evita di proporne altri.
# Nella seconda query, non ci sono le "frecce", significa che
# la sto eseguendo in entrambe le direzioni. Mentre nella 
# prima la freccia è verso topic, perché mi serve trovare
# tutti i post che fanno riferimento a quel topic.
@traceable(run_type="retriever", name="kg_topic_history")
def kg_topic_history(topic: str) -> str:
    posts_q = """
    MATCH (t:Topic {name: $topic})<-[:COVERS_TOPIC]-(p:Post)
    RETURN p.title AS title, p.category AS category, p.created_at AS created_at
    ORDER BY p.created_at DESC
    """
    related_q = """
    MATCH (t:Topic {name: $topic})-[:RELATED_TO]-(rt:Topic)
    RETURN DISTINCT rt.name AS related
    """
    try:
        resolved = resolve_topic(topic)
        posts = run_read(posts_q, topic=resolved)
        if not posts:
            return f"L'argomento '{topic}' non e' mai stato trattato nel blog (e' un possibile gap di copertura)."

        lines = [f"L'argomento '{topic}' e' gia' stato trattato nei seguenti post:"]
        for r in posts:
            lines.append(f"- {r['title']} (Categoria: {r['category']}, del {fmt_date(r['created_at'])})")

        related = [r["related"] for r in run_read(related_q, topic=resolved) if r["related"]]
        if related:
            lines.append("Topic correlati gia' presenti nel grafo: " + ", ".join(related))
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura di Neo4j: {str(e)}"



# Metodo che torna la panoramica di tutti i topic nel KG, usata anche questa
# dal planner per la gap-analysis.
# In questo caso, uso OPTIONAL MATCH, in cui a differenza del MATCH classico
# non scarto i topic che non hanno post, ma li includo in post_count. E questo
# mi serve proprio per capire i topic senza post e quindi il gap di copertura che
# serve al planner per sapere cosa scrivere.
@traceable(run_type="retriever", name="kg_topics_overview")
def kg_topics_overview() -> str:
    query = """
    MATCH (t:Topic)
    OPTIONAL MATCH (t)<-[:COVERS_TOPIC]-(p:Post)
    RETURN t.name AS topic, count(p) AS post_count, max(p.created_at) AS last_post
    ORDER BY post_count ASC, last_post ASC
    """
    try:
        rows = run_read(query)
        if not rows:
            return "Il Knowledge Graph e' vuoto: nessun topic registrato."

        lines = ["Panoramica della copertura editoriale (dal piu' trascurato):"]
        for r in rows:
            if r["post_count"] == 0:
                lines.append(f"- {r['topic']}: Mai trattato, possibile gap di copertura.")
            else:
                lines.append(
                    f"- {r['topic']}: {r['post_count']} post, ultimo il {fmt_date(r['last_post'])}."
                )
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura di Neo4j: {str(e)}"




# Contesto editoriale completo per la stesura (quindi usato dal drafting_node e research_agent)
# La query raccoglie tutto ciò che il KG sa su uno specifico topic, titoli dei post (per evitare ripetizioni)
# le claims già affermate (per coerenza), le fonti usate (per il riuso o riferimenti tra post) e i topic correlati (per cross-link).
# collect(DISTINCT...) serve pre tornare una lista senza duplicati (se avessi un post con 3 claim e 2 fonti, avrei 6 valori ripetuti).

@traceable(run_type="retriever", name="kg_topic_context")
def kg_topic_context(topic: str) -> str:
    """
    Contesto editoriale completo di un topic per garantire COERENZA e CROSS-LINK
    in fase di stesura: post esistenti, claim chiave gia' affermati, fonti usate,
    topic correlati. E' anche il pezzo di KG che alimenta il K-RAG.
    """
    query = """
    MATCH (t:Topic {name: $topic})
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
        resolved = resolve_topic(topic)
        rows = run_read(query, topic=resolved)
        if not rows:
            return f"Nessun contesto nel KG per '{topic}': e' un argomento nuovo."

        r = rows[0]
        # Limito gli elementi per non gonfiare il kickoff: bastano pochi titoli e claim
        # per la coerenza e i cross-link. Le "fonti gia' usate" NON vengono piu' iniettate:
        # erano il blocco piu' pesante e meno utile (un post nuovo raccoglie le proprie
        # fonti) e nel grafo contengono spesso output grezzi dei tool, non URL puliti.
        posts = [x for x in r["posts"] if x][:3]
        claims = [x for x in r["claims"] if x][:3]
        related = [x for x in r["related"] if x][:5]

        if not any([posts, claims, related]):
            return f"Nessun contesto nel KG per '{topic}': e' un argomento nuovo."

        lines = [f"Contesto del KG per il topic '{topic}' (usalo per coerenza e link interni):"]
        if posts:
            lines.append("Post esistenti collegati: " + "; ".join(posts))
        if claims:
            lines.append("Claim gia' affermati (non contraddirli): " + "; ".join(claims))
        if related:
            lines.append("Topic correlati per cross-link: " + ", ".join(related))
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura di Neo4j: {str(e)}"



# Metodo che torna solo i nomi dei topic correlati, serve per la query expansion del K-RAG
# La query expansion mi serve per ottenere più risultati legati al topic di partenza.
# Prendendo l'esempio dell'aerodinamica attiva, per espanderla chiedo al KG i topic collegati
# espando la query aggiungendo i termini dei topic e poi la eseguo nel RAG.
@traceable(run_type="retriever", name="kg_related_topics")
def kg_related_topics(topic: str) -> list[str]:
    """
    Restituisce SOLO la lista dei topic correlati a un dato topic nel KG.
    Usata per la QUERY EXPANSION del K-RAG: le entita'/topic correlati del KG
    vengono accodati alla query di retrieval per espanderla (requisito:
    "use the Knowledge Graph to expand or refine retrieval queries").
    """
    query = """
    MATCH (t:Topic {name: $topic})-[:RELATED_TO]-(rt:Topic)
    RETURN DISTINCT rt.name AS related
    """
    try:
        resolved = resolve_topic(topic)
        rows = run_read(query, topic=resolved)
        return [r["related"] for r in rows if r["related"]]
    except Exception:
        return []


# Lettura del backlog di proposte (:Proposal) non ancora scritte.
# Usata da suggest_topics_node: quando l'utente chiede "di cosa parliamo oggi", le
# proposte rimaste in sospeso da piani precedenti rientrano in gioco, in ordine di
# creazione (cosi' possono essere recuperate e scritte piu' avanti).
@traceable(run_type="retriever", name="kg_pending_proposals")
def kg_pending_titles_list() -> list:
    """
    Restituisce i SOLI titoli delle proposte pendenti come lista Python ordinata
    (dalla piu' vecchia). Serve al gate dei suggerimenti per mappare la scelta
    dell'utente ("la proposta 2") sul titolo giusto in modo deterministico.
    """
    query = """
    MATCH (pr:Proposal)
    RETURN pr.title AS title
    ORDER BY pr.created_at ASC
    """
    try:
        rows = run_read(query)
        return [r.get("title") or "" for r in rows if r.get("title")]
    except Exception:
        return []


def kg_pending_proposals() -> str:
    """
    Restituisce le proposte editoriali pendenti, formattate, ordinate dalla piu' vecchia.
    Stringa vuota se non ce ne sono (cosi' il chiamante puo' semplicemente non mostrarle).
    """
    query = """
    MATCH (pr:Proposal)
    RETURN pr.title AS title, pr.category AS category,
           pr.justification AS justification, pr.created_at AS created_at
    ORDER BY pr.created_at ASC
    """
    try:
        rows = run_read(query)
        if not rows:
            return ""
        lines = ["Proposte in sospeso da piani precedenti (recuperabili):"]
        for i, r in enumerate(rows, 1):
            cat = r.get("category") or "n/d"
            title = r.get("title") or "(senza titolo)"
            just = r.get("justification") or ""
            lines.append(f"{i}. [{cat}] {title}" + (f"\n   Motivazione: {just}" if just else ""))
        return "\n".join(lines)
    except Exception as e:
        return f"Errore durante la lettura delle proposte: {str(e)}"


# Elenco COMPATTO dei post gia' pubblicati di recente (titoli reali + categoria), usato
# dal planner per la CONTINUITA': quando l'utente chiede un confronto/seguito con qualcosa
# "gia' trattato", il planner deve riferirsi a un modello REALE preso da qui, non inventarne
# uno. Restituisce stringa vuota se non c'e' nulla.
@traceable(run_type="retriever", name="kg_recent_posts")
def kg_recent_posts(limit: int = 8) -> str:
    query = """
    MATCH (p:Post)
    RETURN p.title AS title, p.category AS category, p.created_at AS created_at
    ORDER BY p.created_at DESC
    LIMIT $limit
    """
    try:
        rows = run_read(query, limit=limit)
        if not rows:
            return ""
        lines = []
        for r in rows:
            cat = r.get("category") or "n/d"
            title = r.get("title") or "(senza titolo)"
            lines.append(f"- [{cat}] {title}")
        return "\n".join(lines)
    except Exception as e:
        return f"(errore lettura post pubblicati: {str(e)})"


# Elenco COMPATTO (solo titoli) delle proposte in sospeso, per il planner: cosi' sa cosa
# c'e' gia' in backlog ed evita di riproporlo, potendo invece dargli continuita'.
@traceable(run_type="retriever", name="kg_proposed_titles")
def kg_proposed_titles() -> str:
    query = """
    MATCH (pr:Proposal)
    RETURN pr.title AS title, pr.category AS category
    ORDER BY pr.created_at ASC
    """
    try:
        rows = run_read(query)
        if not rows:
            return ""
        return "\n".join(f"- [{r.get('category') or 'n/d'}] {r.get('title') or ''}" for r in rows)
    except Exception as e:
        return f"(errore lettura proposte: {str(e)})"
