"""
Modulo che implementa il protocollo MCP (Model Context Protocol)
banalmente, serve per permettere all'agente di comunicare con un servizio
esterno. Nel mio caso ho trasformato la ricerca web in un servizio
indipendente tramite l'MCP invece che come servizio interno (e quindi un tool classico).

Si avvia come servizio persistente (in un terminale dedicato) e resta
in ascolto su http://127.0.0.1:8765/mcp. L'agente lo chiama via HTTP a ogni ricerca.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Fase di setup, uso FastMCP come framework standard per l'MCP.


from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from config.settings import Configuration

load_dotenv()
cfg = Configuration()

mcp_server = FastMCP("AutomotiveSearchServer", host="127.0.0.1", port=8765)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

# Parametri di ricerca che posso essere adattati.
# Servono come trade-off tra qualità della ricerca e velocità.
# Impostati così, ogni ricerca web impiega dai 40 secondi fino ai 60.
# Questi rappresentano il compromesso migliore.
MAX_RESULTS = 5
SEARCH_DEPTH = "advanced"   # Advanced da risultati più ricchi, ci sono altri modificatori, come basic per risultati normali
INCLUDE_RAW = True          # Includo il testo integrale, scartando elementi irrilevanti della pagina.

# Whitelist dei domini in cui cercare. Spesso la ricerca
# tornava "fonti" poco attendibili (es. video di YouTube, social, ecc.)
# quindi ho costruito questa lista da cui prendere le fonti.

ALLOWED_DOMAINS = [
    "quattroruote.it", "alvolante.it", "automoto.it", "motori.it",
    "ansa.it", "autoblog.it", "omniauto.it", "hdmotori.it",
    "dueruote.it", "moto.it", "insella.it", "motociclismo.it",
    "motorbox.com", "automoto.it", "sicurauto.it", "vaielettrico.it",
    "formulapassion.it",
]


# Configurazione del modello che riassume il contenuto delle pagine.
# Usavo Phi 4 Mini in passato, ma l'ho confrontato con Ministral 3,
# ed è molto più veloce e anche più discorsivo.
summarizer_llm = ChatOllama(
    model=cfg.summarizer_model_name,
    num_ctx=cfg.summarizer_num_ctx,
    temperature=cfg.summarizer_temperature,
)

# Prompt per la sintesi di una fonte/articolo.
SUMMARIZE_ONE_PROMPT = """Sei un assistente che riassume una pagina web per un blog automotive.
Riassumi il contenuto in italiano in 4-6 frasi complete, mantenendo TUTTI i dati concreti
(modelli, cilindrate, potenze, prezzi, date, luoghi, dichiarazioni). Riporta le unita' di misura
ESATTAMENTE come nell'originale (NON convertire lb-ft, mph, libbre, ecc.). NON inventare nulla
che non sia nel testo. Scrivi solo il riassunto, senza preamboli."""



# Metodo che riassume un singolo articolo, se è breve lo lascio così
# se il riassunto dovesse non funzionare per qualche motivo
# torno i primi 800 caratteri (sperando che contengano informazioni utili).
def _summarize_one(title: str, content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    if len(text.split()) < 30:
        return text 
    try:
        resp = summarizer_llm.invoke([
            SystemMessage(content=SUMMARIZE_ONE_PROMPT),
            HumanMessage(content=f"Titolo: {title}\n\nContenuto:\n{text[:6000]}"),
        ])
        return (resp.content or "").strip() or text[:800]
    except Exception:
        return text[:800]


# Metodo che toglie i duplicati tra i risultati, tenendo solo il primo URL che compare.
# Spesso Tavily li torna.
def _deduplicate(results: list) -> list:
    seen, unique = set(), []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)
    return unique

# Metodo che definisce il formato in output atteso per ogni fonte. 
# La formattazione di ogni riassunto deve essere: Fonte N, Titolo, Riassunto, URL e la data
# ma spesso non c'è perché Tavily fa le ricerche di default come topic "general"
# nelle "news" invece compare.
def _format_output(processed: list) -> str:
    if not processed:
        return "Nessun risultato valido dalla ricerca web. Prova con una query diversa."
    blocks = ["Risultati della ricerca web:"]
    for i, r in enumerate(processed, 1):
        block = (
            f"\nFONTE {i}:\n"
            f"Titolo: {r['title']}\n"
            f"Riassunto: {r['summary']}\n"
            f"URL: {r['url']}"
        )
        if r.get("date"):
            block += f"\nData pubblicazione: {r['date']}"
        blocks.append(block)
    return "\n".join(blocks)


# Dizionario di parole chiave per la ricerca di attualità (usato per decidere se usare topic="news" o topic="general")
_NEWS_KEYWORDS = (
    "novità", "novita", "salone", "fiera", "presentat", "svelat", "lancio", "lanciat",
    "ultime", "ultimo", "ultima", "annuncio", "annunciat", "debutto", "debutta",
    "in arrivo", "2026", "2027", "notizie", "news", "evento", "anteprima",
)



# Metodo che sfrutta il dizionario sopra per scegliere il topic
def _pick_topic(query: str) -> str:
    q = (query or "").lower()
    return "news" if any(k in q for k in _NEWS_KEYWORDS) else "general"

# Metodo che scarta risultati non validi, come sitemap, file XML e pagine senza contenuti (solo elenco di URLs).
# Evito che il modello che riassume, si confonda e torni un output illeggibile.
def _is_valid_article(r: dict) -> bool:

    url = (r.get("url") or "").lower()
    title = (r.get("title") or "").lower()
    content = r.get("content") or ""
    # URL che non sono articoli
    bad_markers = ("sitemap", ".xml", "/feed", "rss", "/tag/", "/category/", "/categoria/")
    if any(m in url for m in bad_markers):
        return False
    if "[xml]" in title or "sitemap" in title:
        return False
    # Contenuto troppo povero o che sembra una lista di URL (molti "http" = indice/sitemap)
    if content.count("http") > 5:
        return False
    if len(content.strip()) < 80:
        return False
    return True

# Filtro che banalmente tiene in considerazione le parole chiave della query, se
# la fonte non contiene nessuna di queste allora viene scartata.
# Se nessuna delle fonti ha le keyword, non applico il filtro.
def _relevance_filter(query: str, results: list) -> list:
    import re
    stop = {"recensioni", "recensione", "prestazioni", "affidabilità", "affidabilita",
            "design", "review", "comparison", "confronto", "ultime", "novità", "novita",
            "performance", "reliability", "specifiche", "scheda", "tecnica", "tecnico",
            "innovazione", "tecnologica", "sportività", "sportivita", "praticità", "praticita",
            "2024", "2025", "2026", "2027", "con", "and", "the", "una", "uno", "della", "delle"}
    keywords = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3 and w.lower() not in stop]
    if not keywords:
        return results
    kept = []
    for r in results:
        hay = (str(r.get("title", "")) + " " + str(r.get("content", ""))).lower()
        if any(k in hay for k in keywords):
            kept.append(r)
    return kept if kept else results

# Metodo che esegue la logica MCP. L'agente fa una query, dal topic capisco se è una news o no,
# viene eseguita la ricerca tramite Tavily, deduplico i risultati, scarto i 
# i non articoli, applico il filtro di rilevanza sopra, per ogni risultato ottenuto
# faccio un riassunto e poi formatto l'output in modo da essere leggibile sia
# dall'agente che da me su LangSmith.
@mcp_server.tool()
def search_and_summarize(query: str) -> str:
    try:
        topic = _pick_topic(query)
        result = tavily_client.search(
            query=query,
            search_depth=SEARCH_DEPTH,
            max_results=MAX_RESULTS,
            include_raw_content=INCLUDE_RAW,
            include_domains=ALLOWED_DOMAINS,
            topic=topic,
        )
        results = result.get("results", []) if isinstance(result, dict) else []
        results = _deduplicate(results)

        # Nel caso in cui la whitelist non faccia tornare nessuna fonte
        # faccio si che la ricerca venga fatta globalmente in tutto il web
        # meglio avere qualcosa (anche di sbagliato) che niente completamente.
        if not results:
            result = tavily_client.search(
                query=query, search_depth=SEARCH_DEPTH, max_results=MAX_RESULTS,
                include_raw_content=INCLUDE_RAW, include_domains=ALLOWED_DOMAINS, topic="general",
            )
            results = _deduplicate(result.get("results", []) if isinstance(result, dict) else [])

        if not results:
            return "Nessuna informazione pertinente trovata sul web per questa query."

        results = [r for r in results if _is_valid_article(r)]
        if not results:
            return "Nessuna fonte web valida trovata per questa query (solo pagine non-articolo)."


        results = _relevance_filter(query, results)

        processed = []
        for r in results:
            title = r.get("title") or r.get("url") or "Fonte web"
            url = r.get("url") or ""
            date = r.get("published_date") or ""
            content = (r.get("raw_content") if INCLUDE_RAW else None) or r.get("content") or ""
            summary = _summarize_one(title, content)
            if summary:
                processed.append({"title": title, "url": url, "summary": summary, "date": date})

        return _format_output(processed)

    except Exception as e:
        return f"Errore nel Server MCP durante la ricerca o la sintesi: {str(e)}"


if __name__ == "__main__":
    print("Server MCP avviato su http://127.0.0.1:8765/mcp", file=sys.stderr)
    mcp_server.run(transport="streamable-http")
