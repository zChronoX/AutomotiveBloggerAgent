"""
Tool che recupera le ultime notizie dai feed RSS di testate automotive italiane.
L'agente lo usa quando chiedo suggerimenti o spunti per i prossimi post (insieme alla gap-analysis del KG).
Uso piu' feed per avere piu' varieta' di notizie; se un feed non risponde, continuo con gli altri.
"""

import feedparser
from langchain_core.tools import tool
from prompts.tool_prompts import TREND_ANALYSIS_PROMPT


# Feed RSS di testate automotive italiane, verificati come funzionanti.
# Piu' fonti = piu' varieta' di trend. Se una non risponde, le altre coprono comunque.
RSS_FEEDS = [
    "https://it.motor1.com/rss/news/all/",
    "https://www.automoto.it/rss/news.xml",
    "https://www.hdmotori.it/feed/",
]

MAX_TITOLI = 8  # quanti titoli totali restituire


@tool(description=TREND_ANALYSIS_PROMPT)
def fetch_automotive_trends() -> str:
    titoli = []
    feed_falliti = 0

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            # bozo + nessuna entry = feed rotto/irraggiungibile: lo salto, provo gli altri.
            # (bozo da solo non basta: alcuni feed validi danno warning non fatali.)
            if getattr(feed, "bozo", 0) and not feed.entries:
                feed_falliti += 1
                continue
            for entry in feed.entries:
                titolo = getattr(entry, "title", "").strip()
                if titolo and titolo not in titoli:   # evito doppioni tra feed diversi
                    titoli.append(titolo)
        except Exception:
            feed_falliti += 1
            continue

    # Se TUTTI i feed sono falliti, segnalo il problema (diverso da "nessuna notizia").
    if not titoli:
        if feed_falliti == len(RSS_FEEDS):
            return "Impossibile leggere i feed dei trend automotive (tutte le fonti irraggiungibili)."
        return "Nessun trend trovato al momento nei feed."

    results = "Ultimi trend e notizie Automotive di oggi:\n"
    for i, titolo in enumerate(titoli[:MAX_TITOLI], 1):
        results += f"{i}. {titolo}\n"
    return results
