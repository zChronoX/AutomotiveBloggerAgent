"""
Tool che aggiorna/scrive nel Knoledge Graph, viene chiamata solo dopo l'approvazione dell'utente, mai dall'agente in autonomia.
Quindi sfrutta l'HITL.
"""

from typing import List, Optional
from .client import get_db_driver, open_session




# Funzione che applica internamente il metodo best_semantic_match, 
# in modo da usare il topic esistente se il confronto
# via embeddings è superiore della soglia impostata.
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
    # Risoluzione semantica del topic.
    try:
        from .queries import resolve_topic
        topic = resolve_topic(topic)
    except Exception:
        # Se la risoluzione fallisce, usiamo il topic canonico cosi' com'e'.
        topic = (topic or "").lower().strip()


    # Costruisco la query Cypher. Il merge cerca un nodo
    # con etichetta topic e proprità name, uguale a $topic,
    # se lo trova lo riusa, altrimenti lo crea, così evito duplicati.
    # normalizzo a minuscolo il titolo direttamente con Neo4J.

    # ON CREATE SET viene eseguito solo se il nodo è stato appena creato
    # altrimenti viene eseguito ON MATCH SET.

    #Il MERGE alla fine collega il post e il topic, evitando duplicati come sopra.

    # FOREACH itera sugli elementi della lista $related_topics,
    # così crea/trova un topic con quel nome, e crea una relazione
    # dal topic principale a quello correlato.
    # Ottengo una query expansion, quando l'agente cercherà "aerodinamica attiva"
    # troverà anche "prestazioni" e "hypercar".

    # Per ogni fonte, si crea un nodo Source, collegato al post con BASED_ON, 
    # se due post hanno come riferimento quella fonte, puntano allo stesso nodo.

    # Le claims, sono le affermazioni chiave dei post, prese da "update_kg_node"
    # Ogni claim è un nodo che il "drafting_node" usa per coerenza
    # se sta scrivendo post su uno stesso tema, lo consulta e capisce che non
    # deve scrivere cose incoerenti.


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


    # Creo una connessione a Neo4J, apro una sessione, eseguo la query Cypher passando 
    # tutti i parametri che servono, ritorna una stringa di conferma, o un errore.
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



# Proposte editoriali
# Quando l'utente sceglie quali post scrivere, l'intera selezione viene salvata
# subito come nodi :Proposal (crash-safe): se l'agente si interrompe a metà, i post
# non ancora scritti non vanno persi. Ogni post pubblicato con successo viene poi
# rimosso dalle proposte (vedi remove_proposal, chiamata in update_kg_node).
#
# La :Proposal tiene il topic come proprietà (topic_key) e non crea
# nodi Topic né relazioni. Cosi' la gap-analysis (che conta i :Post via COVERS_TOPIC)
# non viene inquinata da proposte non ancora scritte. Il campo 'title' resta leggibile
# per la presentazione; 'topic_key' e' la chiave canonica per il match/dedup/rimozione.
def add_proposals(proposals: List[dict]) -> str:
    """
    Salva (o aggiorna) una lista di proposte editoriali come nodi :Proposal.
    Ogni elemento e' un dict con: title, topic_key, category, justification.
    Il MERGE su topic_key evita duplicati se lo stesso tema viene riproposto.
    """
    if not proposals:
        return "Nessuna proposta da salvare."

    query = """
    UNWIND $items AS item
    MERGE (pr:Proposal {topic_key: item.topic_key})
    ON CREATE SET pr.title = item.title, pr.category = item.category,
                  pr.justification = item.justification, pr.created_at = datetime()
    ON MATCH  SET pr.title = item.title, pr.category = item.category,
                  pr.justification = item.justification
    """
    try:
        driver = get_db_driver()
        with open_session(driver) as session:
            session.run(query, items=proposals)
        return f"Salvate {len(proposals)} proposte editoriali nel Knowledge Graph."
    except Exception as e:
        return f"Errore durante il salvataggio delle proposte: {str(e)}"
    finally:
        if "driver" in locals():
            driver.close()


def remove_proposal(topic_key: str) -> str:
    """
    Rimuove una proposta dal backlog (tipicamente perche' il post e' stato pubblicato
    e quindi 'promosso' a :Post). Usa la chiave canonica topic_key.
    """
    if not topic_key:
        return "Nessuna proposta da rimuovere (topic_key vuoto)."

    query = "MATCH (pr:Proposal {topic_key: $k}) DETACH DELETE pr"
    try:
        driver = get_db_driver()
        with open_session(driver) as session:
            session.run(query, k=topic_key)
        return f"Proposta '{topic_key}' rimossa dal backlog."
    except Exception as e:
        return f"Errore durante la rimozione della proposta: {str(e)}"
    finally:
        if "driver" in locals():
            driver.close()
