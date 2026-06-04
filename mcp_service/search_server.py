"""
Server MCP di ricerca web (transport HTTP streamable).

Si avvia UNA VOLTA come servizio persistente (in un terminale dedicato) e resta
in ascolto su http://127.0.0.1:8765/mcp. L'agente lo chiama via HTTP a ogni ricerca.

Uso (dalla radice del progetto):
    python -m mcp_server.search_server

Design:
- TavilyClient diretto con search_depth="advanced" e whitelist di domini automotive
  affidabili (riduce rumore: niente video YouTube o fonti casuali);
- sintesi PER-ARTICOLO con il modello locale (riassunto sostanzioso, non una frase);
- output FORMATTATO leggibile: per ogni fonte Titolo / Riassunto / URL / Data.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

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

# ------------------------------------------------------------
# PARAMETRI DI RICERCA (tarabili in cima al file)
# ------------------------------------------------------------
MAX_RESULTS = 5
SEARCH_DEPTH = "advanced"   # "advanced" = risultati piu' ricchi (consigliato), "basic" = piu' veloce
INCLUDE_RAW = True          # includi il testo integrale: da' al riassuntore piu' materiale

# Whitelist di domini automotive/news italiani affidabili: limita il rumore (no YouTube,
# social, ecc.) e mantiene le fonti in italiano, coerenti con un blog italiano.
# Aggiungi/togli domini secondo le tue fonti di fiducia.
ALLOWED_DOMAINS = [
    "quattroruote.it", "alvolante.it", "automoto.it", "motori.it",
    "ansa.it", "ilsole24ore.com", "corriere.it", "repubblica.it",
    "autoblog.it", "omniauto.it", "everyeye.it", "hdmotori.it",
    "dueruote.it", "moto.it", "insella.it", "motociclismo.it",
    "motorbox.com", "automoto.it", "sicurauto.it", "vaielettrico.it",
    "formulapassion.it", "gazzetta.it",
]

summarizer_llm = ChatOllama(
    model=cfg.summarizer_model_name,
    num_ctx=cfg.summarizer_num_ctx,
    temperature=cfg.summarizer_temperature,
)

# Prompt di sintesi PER-ARTICOLO. Riassunto SOSTANZIOSO (non una frase): vogliamo che il
# modello scrittore abbia abbastanza materiale. Vincolo: NON convertire le unita' di misura.
SUMMARIZE_ONE_PROMPT = """Sei un assistente che riassume una pagina web per un blog automotive.
Riassumi il contenuto in italiano in 4-6 frasi complete, mantenendo TUTTI i dati concreti
(modelli, cilindrate, potenze, prezzi, date, luoghi, dichiarazioni). Riporta le unita' di misura
ESATTAMENTE come nell'originale (NON convertire lb-ft, mph, libbre, ecc.). NON inventare nulla
che non sia nel testo. Scrivi solo il riassunto, senza preamboli."""


def _summarize_one(title: str, content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    if len(text.split()) < 30:
        return text  # gia' breve: lascialo com'e'
    try:
        resp = summarizer_llm.invoke([
            SystemMessage(content=SUMMARIZE_ONE_PROMPT),
            HumanMessage(content=f"Titolo: {title}\n\nContenuto:\n{text[:6000]}"),
        ])
        return (resp.content or "").strip() or text[:800]
    except Exception:
        return text[:800]


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


def _format_output(processed: list) -> str:
    """Formato richiesto: per ogni fonte Titolo / Riassunto / URL / Data (se disponibile)."""
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
        # La data compare solo se Tavily l'ha effettivamente restituita (con topic='general'
        # spesso non c'e'): meglio omettere la riga che scrivere sempre 'non disponibile'.
        if r.get("date"):
            block += f"\nData pubblicazione: {r['date']}"
        blocks.append(block)
    return "\n".join(blocks)


# Parole che segnalano una richiesta di ATTUALITA' (notizia/evento recente): per queste
# usiamo topic="news" su Tavily, che restituisce risultati recenti CON data di pubblicazione
# (utile per privilegiare le fonti piu' aggiornate). Per tutto il resto (specifiche tecniche,
# recensioni, confronti) usiamo "general". Dizionario deterministico e piccolo: niente modello.
_NEWS_KEYWORDS = (
    "novità", "novita", "salone", "fiera", "presentat", "svelat", "lancio", "lanciat",
    "ultime", "ultimo", "ultima", "annuncio", "annunciat", "debutto", "debutta",
    "in arrivo", "2026", "2027", "notizie", "news", "evento", "anteprima",
)


def _pick_topic(query: str) -> str:
    """Sceglie il topic Tavily: 'news' per l'attualita' (con data), 'general' altrimenti."""
    q = (query or "").lower()
    return "news" if any(k in q for k in _NEWS_KEYWORDS) else "general"


def _is_valid_article(r: dict) -> bool:
    """
    Scarta i risultati che non sono articoli leggibili: sitemap, file XML/feed, pagine di
    categoria senza contenuto reale. Tavily a volte li restituisce e il riassuntore non puo'
    ricavarne nulla di sensato (producono blob di URL illeggibili).
    """
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


def _relevance_filter(query: str, results: list) -> list:
    """
    Filtro di rilevanza minimo: tiene solo i risultati il cui titolo/contenuto contiene
    almeno una delle parole 'forti' della query (marca, modello, termini chiave > 3 lettere).
    Serve a scartare i risultati totalmente fuori tema che Tavily a volte restituisce, specie
    con topic='news' dove privilegia la freschezza sulla pertinenza (es. una Bentley su una
    query Honda). Se nessun risultato contiene le keyword, non filtriamo (meglio qualcosa che
    niente, e il Self-RAG a valle fara' da secondo controllo).
    """
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


@mcp_server.tool()
def search_and_summarize(query: str) -> str:
    """
    Esegue una ricerca web automotive (Tavily, ricerca avanzata su domini affidabili) e
    riassume OGNI fonte con il modello locale. Restituisce un output formattato e leggibile:
    per ciascuna fonte Titolo, Riassunto, URL e (se disponibile) Data di pubblicazione.
    Per le query di attualita' usa topic='news' (risultati recenti con data), altrimenti 'general'.
    """
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

        # Fallback: se la whitelist non trova nulla, riproviamo UNA volta mantenendo la
        # whitelist ma con topic='general' (piu' legato al contenuto della query che alla
        # freschezza). NON ripetiamo senza whitelist: senza domini, con news, Tavily pescava
        # notizie automotive casuali (es. una Bentley su una query Honda).
        if not results:
            result = tavily_client.search(
                query=query, search_depth=SEARCH_DEPTH, max_results=MAX_RESULTS,
                include_raw_content=INCLUDE_RAW, include_domains=ALLOWED_DOMAINS, topic="general",
            )
            results = _deduplicate(result.get("results", []) if isinstance(result, dict) else [])

        if not results:
            return "Nessuna informazione pertinente trovata sul web per questa query."

        # Scarta sitemap, feed, pagine di categoria (non sono articoli leggibili).
        results = [r for r in results if _is_valid_article(r)]
        if not results:
            return "Nessuna fonte web valida trovata per questa query (solo pagine non-articolo)."

        # Filtro di rilevanza: scarta i risultati fuori tema (marca/modello assenti dal titolo).
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
    print("[MCP SERVER] Avvio su http://127.0.0.1:8765/mcp ...", file=sys.stderr)
    print("[MCP SERVER] Lascia questa finestra aperta. CTRL+C per fermare.", file=sys.stderr)
    mcp_server.run(transport="streamable-http")
