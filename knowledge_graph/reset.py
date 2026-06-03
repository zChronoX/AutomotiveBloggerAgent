"""
Svuota COMPLETAMENTE il Knowledge Graph (Neo4j).

ATTENZIONE: operazione IRREVERSIBILE. Cancella TUTTI i nodi e TUTTE le relazioni del
database Neo4j configurato nel .env. Da usare per ripartire da un grafo vuoto (es. prima
di una demo pulita, da rilanciare poi seed_kg.py se si vogliono i post-esempio).

Richiede una conferma esplicita da tastiera per evitare cancellazioni accidentali.

Uso (dalla radice del progetto, con Neo4j attivo e .env configurato):
    python -m knowledge_graph.reset
"""

from dotenv import load_dotenv
load_dotenv()

from .client import get_db_driver, open_session


def main():
    print("== RESET del Knowledge Graph (Neo4j) ==")
    print("ATTENZIONE: questa operazione cancella TUTTI i nodi e le relazioni del database.")
    print("E' IRREVERSIBILE.\n")
    conferma = input("Per procedere scrivi esattamente 'CANCELLA TUTTO': ").strip()

    if conferma != "CANCELLA TUTTO":
        print("Operazione annullata: nessuna modifica al database.")
        return

    driver = get_db_driver()
    try:
        with open_session(driver) as session:
            # Conta prima, per dare un feedback utile
            before = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            # DETACH DELETE rimuove nodi E relazioni in un colpo solo
            session.run("MATCH (n) DETACH DELETE n")
            after = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(f"\nFatto. Nodi prima: {before}, nodi dopo: {after}.")
        print("Il Knowledge Graph e' ora vuoto. Per ripopolarlo con i post-esempio: python -m knowledge_graph.seed")
    except Exception as e:
        print(f"Errore durante il reset: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
