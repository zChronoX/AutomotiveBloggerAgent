"""
Client Neo4j: connessione, gestione sessioni e utility di query.
Codice estratto da kg_tool.py (sezione CONNESSIONE).
"""

import os
import logging
from neo4j import GraphDatabase

# Rete di sicurezza: il driver Neo4j emette le notifiche informative
# (es. "label does not exist" su grafo vuoto) tramite il proprio logger.
# Test diretto (diag_warnings.py) ha confermato che alzare il livello del logger
# a CRITICAL le silenzia. Lo combiniamo coi parametri notifications_min_severity
# su driver e sessione (anch'essi verificati) per ridondanza totale.
for _ln in ("neo4j", "neo4j.notifications", "neo4j.io", "neo4j.pool"):
    logging.getLogger(_ln).setLevel(logging.CRITICAL)
    logging.getLogger(_ln).propagate = False


def get_db_driver():
    """Inizializza la connessione a Neo4j usando le variabili d'ambiente."""
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError("Credenziali Neo4j mancanti nel file .env")

    # Proviamo a disattivare le notifiche gia' a livello driver. Il nome del parametro
    # cambia tra versioni del driver: proviamo le varianti note e, se nessuna e'
    # supportata, ripieghiamo sul driver semplice (le notifiche verranno comunque
    # silenziate a livello di sessione da open_session()).
    for kwargs in (
        {"warn_notification_severity": "OFF"},
        {"notifications_min_severity": "OFF"},
        {"notifications_disabled_categories": ["UNRECOGNIZED", "HINT", "GENERIC"]},
    ):
        try:
            return GraphDatabase.driver(uri, auth=(user, password), **kwargs)
        except (TypeError, ValueError):
            continue
    return GraphDatabase.driver(uri, auth=(user, password))


def open_session(driver):
    """Apre una sessione disattivando, se possibile, le notifiche informative del DBMS
    (es. 'label does not exist' su grafo vuoto). Robusto rispetto alla versione del driver."""
    for kwargs in (
        {"notifications_min_severity": "OFF"},
        {"notifications_disabled_categories": ["UNRECOGNIZED", "HINT", "GENERIC"]},
    ):
        try:
            return driver.session(**kwargs)
        except (TypeError, ValueError):
            continue
    return driver.session()


def run_read(query: str, **params):
    """Esegue una query di lettura e restituisce la lista dei record. Chiude sempre il driver."""
    driver = get_db_driver()
    try:
        with open_session(driver) as session:
            return list(session.run(query, **params))
    finally:
        driver.close()


def fmt_date(value) -> str:
    """Formatta una data Neo4j (o None) in 'YYYY-MM-DD', con fallback robusto."""
    if value is None:
        return "data sconosciuta"
    try:
        # neo4j.time.DateTime -> datetime nativo
        return value.to_native().strftime("%Y-%m-%d")
    except Exception:
        return str(value)
