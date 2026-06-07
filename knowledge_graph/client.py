"""
Modulo di infrastruttura per la gestione del database Neo4j tramite AuraDB.
Gestisce la connessione, apre sessioni ed esegue query.
"""

import os
import logging
from neo4j import GraphDatabase

# Questo blocco di codice mi serve per silenziare le notifiche non importanti che mi manda Neo4j
# quando eseguo query su grafi vuoti, per evitare di intasare la console, ho alzato
# il livello di log a critical e ho impostato propagate a false per evitare che le notifiche vengano stampate.
for _ln in ("neo4j", "neo4j.notifications", "neo4j.io", "neo4j.pool"):
    logging.getLogger(_ln).setLevel(logging.CRITICAL)
    logging.getLogger(_ln).propagate = False




# Creo il diver Neo4J tramite le variabili d'amiente nell'env. 
def get_db_driver():
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError("Credenziali Neo4j mancanti nel file .env")

    # Altra parte di silenziamento delle notifiche, provo vari parametri nel caso una versione di neo4j non supporti un parametro specifico.
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



#Stessa logical del driver ma applicata alla sessione. Silenzio anche qui notifiche informative. 
def open_session(driver):
    for kwargs in (
        {"notifications_min_severity": "OFF"},
        {"notifications_disabled_categories": ["UNRECOGNIZED", "HINT", "GENERIC"]},
    ):
        try:
            return driver.session(**kwargs)
        except (TypeError, ValueError):
            continue
    return driver.session()



#Creo un driver, apro la sessione, eseguo la lettura e chiudo la connessione.
def run_read(query: str, **params):
    driver = get_db_driver()
    try:
        with open_session(driver) as session:
            return list(session.run(query, **params))
    finally:
        driver.close()

#Funzione d'utilità che formatta la data in modo da essere leggibile.
def fmt_date(value) -> str:
    if value is None:
        return "data sconosciuta"
    try:
        return value.to_native().strftime("%Y-%m-%d")
    except Exception:
        return str(value)
