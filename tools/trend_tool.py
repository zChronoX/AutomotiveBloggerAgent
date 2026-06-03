import feedparser
from langchain_core.tools import tool
from prompts.tool_prompts import TREND_ANALYSIS_PROMPT


@tool(description=TREND_ANALYSIS_PROMPT)
def fetch_automotive_trends() -> str:
    """Legge il feed RSS automotive (Motor1) e restituisce i titoli piu' recenti."""
    url = "https://it.motor1.com/rss/news/all/"
    try:
        feed = feedparser.parse(url)

        # feedparser NON solleva eccezioni sui feed irraggiungibili/malformati:
        # segnala il problema in feed.bozo. Lo controlliamo per distinguere
        # "nessuna notizia" da "feed non raggiungibile".
        if getattr(feed, "bozo", 0) and not feed.entries:
            motivo = getattr(feed, "bozo_exception", "feed non valido o non raggiungibile")
            return f"Impossibile leggere il feed dei trend automotive: {motivo}."

        if not feed.entries:
            return "Nessun trend trovato al momento nei feed."

        results = "Ultimi trend e notizie Automotive di oggi:\n"
        for i, entry in enumerate(feed.entries[:5]):
            results += f"{i+1}. {entry.title}\n"
        return results

    except Exception as e:
        return f"Errore nel recupero dei trend: {str(e)}"