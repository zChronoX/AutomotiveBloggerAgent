"""
Utility interne del modulo agent.
Funzioni helper pure estratte da blogger_agent.py.
"""

import re
from langchain_core.messages import ToolMessage


# ============================================================
# REASONING TRACE
# ============================================================
def trace(state: dict, line: str) -> str:
    """Aggiunge una riga al reasoning trace nello stato."""
    prev = state.get("reasoning_trace") or ""
    return (prev + "\n" + line).strip()


# ============================================================
# INTENT DETECTION (suggerimento vs scrittura)
# ============================================================
_SUGGEST_WORDS = [
    "suggerisci", "suggeriscimi", "proponi", "idee", "argomenti",
    "spunti", "di cosa", "consigli", "consigliami", "brainstorm",
]

_WRITE_WORDS = [
    "scrivi", "bozza", "articolo", "redigi", "redarre", "stesura",
    "prepara un post", "scrivimi", "componi", "genera un post",
    "in un post", "un post su", "un post sul", "un post sulla",
    "confronta", "confronto", "metti a confronto", "paragona",
    "recensione", "recensisci", "parla di", "parlami di", "parlami",
    "parlaci", "guida su", "guida sulla", "guida sui", "fammi un post",
]


def wants_post(text: str) -> bool:
    """
    True se l'utente vuole SCRIVERE un post; False se vuole solo SUGGERIMENTI.

    I termini di suggerimento hanno PRIORITA': frasi come "suggeriscimi argomenti per i
    prossimi post" contengono 'post' ma NON sono richieste di scrittura.
    """
    t = (text or "").lower()
    if any(w in t for w in _SUGGEST_WORDS):
        return False
    return any(w in t for w in _WRITE_WORDS)


# ============================================================
# SOURCE VALIDATION (guardrail fonti)
# ============================================================
# Tool le cui osservazioni contano come "fonte di contenuto" sufficiente a NON forzare
# la ricerca web. NOTA: fetch_automotive_trends (feed RSS) e' ESCLUSO di proposito: e'
# una lista di titoli di tendenza, non contenuto reale su un tema specifico. Se il modello
# ha solo i trend, il guardrail deve comunque spingerlo a cercare sul web (es. un evento
# di attualita' come un salone richiede una ricerca vera, non i titoli RSS).
_GROUNDING_TOOLS = {
    "retrieve_local_documents", "mcp_web_search",
    "fetch_vehicle_specs", "compare_vehicles",
}


def has_collected_sources(state: dict) -> bool:
    """
    True se nella cronologia c'e' almeno UNA fonte valida raccolta dai tool di
    ricerca/grounding (un ToolMessage non vuoto e senza errore). Serve al guardrail
    "verifica fonti": un post non deve essere scritto basandosi solo sulla conoscenza
    interna del modello (requisito di progetto: verificare le fonti).
    """
    for m in state.get("messages", []):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") in _GROUNDING_TOOLS:
            content = str(getattr(m, "content", "") or "")
            if (content.strip()
                    and "errore" not in content.lower()[:40]
                    and "not found" not in content.lower()):
                return True
    return False


# ============================================================
# DRAFT CLEANUP
# ============================================================
def strip_reasoning_preamble(text: str) -> str:
    """
    Rimuove un eventuale preambolo di ragionamento (Thought:/Action:/Observation: e simili)
    che alcuni modelli antepongono all'articolo. Conservativo: agisce SOLO se rileva i
    marcatori di ragionamento E c'e' un titolo Markdown dopo; altrimenti restituisce il testo
    invariato (per non tagliare per errore articoli legittimi).
    """
    if not text:
        return text
    lowered = text.lower()
    markers = ("thought:", "action:", "observation:", "procedo con")
    stripped = text.lstrip()
    if stripped.startswith("#"):
        return text
    if not any(m in lowered[:500] for m in markers):
        return text
    match = re.search(r"(?m)^#{1,2}\s+\S", text)
    if match:
        return text[match.start():].lstrip()
    return text


# ============================================================
# TOOL ARGS NORMALIZATION
# ============================================================
def normalize_tool_args(name: str, args: dict) -> dict:
    """
    Corregge gli errori tipici dei modelli locali sui nomi dei parametri.
    Due famiglie di tool con parametro diverso:
      - tool di ricerca (web/RAG): vogliono 'query';
      - tool del Knowledge Graph: vogliono 'topic'.
    Il modello a volte usa il nome dell'altra famiglia (o un sinonimo). Qui rimappiamo
    verso il parametro corretto per il tool chiamato, evitando l'errore "Field required".
    """
    if not isinstance(args, dict):
        return args

    aliases = (
        "query", "text", "target_keyword", "topic", "q",
        "search_query", "question", "input", "argument", "car_model", "keyword",
    )
    query_tools = {"mcp_web_search", "retrieve_local_documents"}
    topic_tools = {"query_knowledge_graph", "get_editorial_context"}

    def _remap(target: str) -> dict:
        if target in args and isinstance(args.get(target), str) and args[target].strip():
            return args
        for alias in aliases:
            if alias == target:
                continue
            if alias in args and isinstance(args[alias], str) and args[alias].strip():
                fixed = dict(args)
                fixed[target] = fixed.pop(alias)
                print(f"[Tool] Argomento '{alias}' rimappato su '{target}' per '{name}'.")
                return fixed
        return args

    if name in query_tools:
        return _remap("query")
    if name in topic_tools:
        return _remap("topic")
    return args


# Parole chiave che, in una richiesta di MODIFICA (HITL), segnalano che servono
# NUOVI DATI -> la modifica deve ripassare dal research agent (puo' chiamare i tool).
# Se nessuna compare, la modifica e' puramente testuale -> resta sul drafting (veloce).
_MODIFICATION_NEEDS_RESEARCH = {
    "aggiungi", "aggiungere", "confronta", "confronto", "paragona", "paragone",
    "cerca", "ricerca", "trova", "includi", "includere", "dati su", "specifiche",
    "informazioni su", "approfondisci", "approfondire", "inserisci un confronto",
    "verifica", "fonti", "fonte", "prezzo", "prezzi", "scheda tecnica",
    "metti a confronto", "compara",
}


def modification_needs_research(feedback: str) -> bool:
    """
    True se la richiesta di modifica dell'utente sembra richiedere NUOVI DATI
    (e quindi un altro giro di ricerca), False se e' una modifica solo testuale
    (accorcia, cambia tono, riscrivi l'introduzione, ecc.).

    Euristica volutamente semplice e prudente: in caso di dubbio (nessuna keyword)
    resta sul drafting, che e' il comportamento attuale gia' testato. Cosi' un
    eventuale falso negativo non peggiora nulla rispetto a prima; solo i casi
    chiari ("aggiungi un confronto con...") attivano la ricerca aggiuntiva.
    """
    if not feedback:
        return False
    low = feedback.lower()
    return any(kw in low for kw in _MODIFICATION_NEEDS_RESEARCH)


# Pattern di richieste PALESEMENTE generiche: su queste forziamo il chiarimento a
# prescindere dal giudizio del modello (Ministral 3B oscilla troppo sul prompt).
# Rete di sicurezza deterministica per lo scoping (notebook 1 Deep Research).
#
# Due categorie:
# - STANDALONE: vaghe solo se sono (quasi) l'intera richiesta, perche' come prefisso
#   sono invece legittime ("scrivi un post SU X" e' chiarissimo).
# - ALWAYS: vaghe ovunque compaiano, perche' indicano esplicito disinteresse al tema.
_VAGUE_STANDALONE = (
    "scrivimi qualcosa", "scrivi qualcosa", "scrivi un post", "scrivimi un post",
    "scrivi un articolo", "scrivimi un articolo", "dammi un articolo", "dammi un post",
    "parlami di auto", "parlami di moto", "parlami di motori",
    "parlami di una moto", "parlami di una auto", "parlami di un auto",
    "parlami di un moto", "parlami di una macchina", "parlami di un veicolo",
    "scrivi di una moto", "scrivi di una auto", "scrivi di un auto",
    "un post a caso", "un argomento qualsiasi",
)
_VAGUE_ALWAYS = (
    "fai tu", "fai te", "decidi tu", "scrivi tu", "quello che vuoi",
    "qualcosa di interessante", "sorprendimi", "come preferisci", "scegli tu",
)


def is_clearly_vague(user_input: str) -> bool:
    """
    True se la richiesta e' PALESEMENTE generica. Rete di sicurezza deterministica
    per lo scoping: questi casi devono SEMPRE far scattare il chiarimento, senza
    dipendere dal modello 3B.

    Logica: un pattern "standalone" (es. "scrivi un post") rende la richiesta vaga
    SOLO se non c'e' sostanza dopo (cioe' nessun tema specificato). "Scrivi un post
    sulla Giulia" NON e' vago perche' dopo il pattern c'e' un tema. I pattern "always"
    (es. "fai tu") sono sempre vaghi.
    """
    if not user_input or not user_input.strip():
        return True  # input vuoto = massimamente vago

    low = " ".join(user_input.lower().replace("'", " ").split())

    # Pattern "always": disinteresse esplicito al tema -> sempre vago
    for pat in _VAGUE_ALWAYS:
        if pat in low:
            return True

    # Pattern "standalone": vaghi solo se NON seguiti da sostanza.
    # Parole "di collegamento" che non contano come sostanza (su, di, per, un, il...).
    _filler = {"su", "di", "del", "della", "dei", "delle", "sul", "sulla", "sui",
               "sulle", "per", "un", "uno", "una", "il", "la", "lo", "le", "gli",
               "che", "a", "e", "con", "tema", "argomento", "qualcosa"}
    for pat in _VAGUE_STANDALONE:
        idx = low.find(pat)
        if idx != -1:
            # Cosa resta dopo il pattern?
            rest = low[idx + len(pat):].strip()
            rest_words = [w for w in rest.split() if w not in _filler]
            # Se non resta nessuna parola di sostanza, e' vago
            if not rest_words:
                return True

    # Richieste estremamente corte e senza sostanza (<= 3 parole, tutte generiche)
    generic_words = {"scrivi", "scrivimi", "post", "articolo", "auto", "moto",
                     "motori", "qualcosa", "un", "uno", "una", "di", "su", "il", "la"}
    words = low.split()
    if len(words) <= 3 and all(w in generic_words for w in words):
        return True
    return False


# Parole da scartare nel derivare il TOPIC CANONICO (chiave di matching del KG).
# Rimuoviamo articoli, preposizioni, parole editoriali e di taglio, in modo che
# "La Giulia Quadrifoglio: storia, simbolo e eredita' dinastica" e
# "Scrivi un post sulla Alfa Romeo Giulia Quadrifoglio" convergano su una chiave simile.
_TOPIC_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "del", "dello",
    "della", "dei", "degli", "delle", "a", "ad", "da", "in", "con", "su", "sul",
    "sulla", "sui", "sulle", "per", "tra", "fra", "e", "ed", "o", "come", "che",
    "scrivi", "scrivimi", "parlami", "dammi", "post", "articolo", "un", "guida",
    "completa", "dettagliata", "recensione", "confronto", "storia", "simbolo",
    "eredita", "eredità", "dinastica", "significato", "descrizione", "tecnica",
    "approfondimento", "tutto", "sul", "vs", "chiarimento", "informazioni", "generali",
}


def canonical_topic(raw: str) -> str:
    """
    Deriva una CHIAVE DI TOPIC canonica, breve e normalizzata, da una richiesta o
    da un titolo. Serve come chiave di matching nel KG per la gap-analysis: due post
    sullo stesso soggetto devono produrre la STESSA chiave.

    Strategia deterministica e robusta:
    - minuscolo, rimozione punteggiatura;
    - taglio alla prima virgola/due punti (i titoli editoriali mettono il taglio dopo);
    - rimozione di articoli, preposizioni e parole editoriali (taglio/forma del post);
    - si tengono al massimo le prime 4 parole di sostanza (il soggetto: marca+modello).
    Esempi:
      "La Giulia Quadrifoglio: storia, simbolo e eredita' dinastica" -> "giulia quadrifoglio"
      "Scrivi un post sulla Alfa Romeo Giulia Quadrifoglio" -> "alfa romeo giulia quadrifoglio"
    """
    if not raw:
        return ""
    s = raw.lower().replace("'", " ").replace("’", " ")
    # taglia al primo separatore di sottotitolo (i due punti o la prima virgola)
    for sep in (":", " - ", " – "):
        if sep in s:
            s = s.split(sep)[0]
            break
    # rimuovi punteggiatura residua
    for ch in ",.;()[]\"":
        s = s.replace(ch, " ")
    words = [w for w in s.split() if w and w not in _TOPIC_STOPWORDS and len(w) > 1]
    # tieni le prime 4 parole di sostanza (marca + modello + eventuale variante)
    return " ".join(words[:4]).strip()
