"""
Utility deterministiche usate dai nodi e nel routing del grafo.
Servono per aiutare un modello piccolo (come Ministral-3 da 3B)
nei vari nodi del grafo. Aiutano a far sì che le sue decisioni siano corrette.
"""

import re
from langchain_core.messages import ToolMessage


# Funzione che accoda una riga di log al campo "reasoning_trace", semplicemente
# aiuta a capire nel main in che fase siamo e cosa è stato fatto. Prende
# lo stato attuale del grafo e lo aggiunge alla variabile "reasoning_trace"
# così capisco tutti i passaggi che l'agente compie.
def trace(state: dict, line: str) -> str:
    """Aggiunge una riga al reasoning trace nello stato."""
    prev = state.get("reasoning_trace") or ""
    return (prev + "\n" + line).strip()


# Definizioni di parole chiave per stabilire se l'utente vuole suggerimenti o se vuole un post vero e proprio
# I termini di suggerimento hanno priorità rispetto la stesura del post stesso, quindi se scrivo
# "Suggeriscimi argomenti per un post" in automatico il "suggeriscimi" nasconde il termine "post"


# Questo è un dizionario di parole per i suggerimenti, quindi quando l'utente
# non vuole scrivere subito un post o dei post, ma vuole solo un suggerimento
_SUGGEST_WORDS = [
    "suggerisci", "suggeriscimi", "proponi", "idee", "argomenti",
    "spunti", "di cosa", "consigli", "consigliami", "brainstorm",
]

#Questo invece è un dizionario di parole per far sì che l'agente capisca che l'utente vuole 
#scrivere un post. Ad esempio se scrivo "scrivi un post su X" lui capirà che deve scrivere un post.
#Abbiamo messo alla fine dei verbi di pianificazione perché servono durante
#la fase editoriale, in cui chiediamo all'agente di pianificare dei post.
#la proprità comunque è nei suggerimenti, quindi prima si entra in suggest_word
#e poi in write words.
_WRITE_WORDS = [
    "scrivi", "bozza", "articolo", "redigi", "redarre", "stesura",
    "prepara un post", "scrivimi", "componi", "genera un post",
    "in un post", "un post su", "un post sul", "un post sulla",
    "confronta", "confronto", "metti a confronto", "paragona",
    "recensione", "recensisci", "parla di", "parlami di", "parlami",
    "parlaci", "guida su", "guida sulla", "guida sui", "fammi un post",
    "pianifica", "pianificami", "preparami", "prepara",
]

# Funzione che mette in pratica quello che c'è scritto su, controlla
# se l'utente ha chiesto un suggerimento o un post, in base alle parole
# presenti del dizionario.
def wants_post(text: str) -> bool:
    """
    True se l'utente vuole scrivere un post; False se vuole solo suggerimenti.
    """
    t = (text or "").lower()
    if any(w in t for w in _SUGGEST_WORDS):
        return False
    return any(w in t for w in _WRITE_WORDS)


#Altro dizionario in cui mappo numeri scritti in lettere al loro valore numerico.
#Viene usata per estrarre quanti post l'utente vuole che l'agente pianifichi.
_NUM_WORDS = {
    "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
}


def extract_num_posts(text: str, default: int = 3, cap: int = 5) -> int:
    """
    Estrae dinamicamente dalla richiesta dell'utente quanti post pianificare.
    Ad esempio: "pianifica 4 post", "scrivimi 3 articoli", "una sequenza
    di 2", "tre post". Se non trova un numero esplicito usa il default.
    Il risultato è sempre limitato da 1 a 5 per non far esplodere tempi/token:
    è il numero massimo di proposte che il planner generera', sarà poi il gate
    editoriale a far scegliere all'utente quante effettivamente scriverne.
    """
    if not text:
        return default
    low = text.lower()

    # Numeri in cifre vicino a parole come post/articoli/proposte/sequenza.
    patterns = [
        r"(\d+)\s+(?:post|articol|propost|pezz|contenut)",
        r"(?:pianific\w+|prepar\w+|scriv\w+|gener\w+)\s+(\d+)",
        r"sequenza\s+di\s+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            try:
                n = int(m.group(1))
                return max(1, min(cap, n))
            except ValueError:
                pass

    # Numeri scritti a parole (es. "tre post").
    for word, val in _NUM_WORDS.items():
        if re.search(rf"\b{word}\s+(?:post|articol|propost)", low):
            return max(1, min(cap, val))

    return default


# Definizioni di quelli che sono i tool "grounding" cioè che danno contesto al modello.
# Servono per capire quando la ricerca ha fonti, oppure tutto ciò che ha scritto il modello
# è stata un'allcinazione perché non aveva fonti reali. Se il modello usa un tool che non è
# nella lista, allora automaticamente deve scattare una ricerca web forzata per validare 
# ciò che ha scritto.
_GROUNDING_TOOLS = {
    "retrieve_local_documents", "mcp_web_search",
    "fetch_vehicle_specs", "compare_vehicles",
}


#Utlity che controlla se il modello ha raccolto fonti durante il processo di ricerca.
def has_collected_sources(state: dict) -> bool:
    """
    True se nella cronologia c'e' almeno una fonte valida raccolta dai tool di
    ricerca/grounding (un ToolMessage non vuoto e senza errore). Serve al guardrail
    "verifica fonti": un post non deve essere scritto basandosi solo sulla conoscenza
    interna del modello che potrebbe allucinare tranquillamente.
    """
    for m in state.get("messages", []):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") in _GROUNDING_TOOLS:
            content = str(getattr(m, "content", "") or "")
            if (content.strip()
                    and "errore" not in content.lower()[:40]
                    and "not found" not in content.lower()):
                return True
    return False


# Metodo che pulisce l'intera catena del ReACT dalla stesura, evitando che venga incluso nel risultato finale del post.
def strip_reasoning_preamble(text: str) -> str:
    """
    Rimuove un eventuale fase di ragionamento ReAct (Thought:/Action:/Observation)
    che alcuni modelli antepongono all'articolo. Agisce solo se rileva i
    marcatori di ragionamento e se c'e' un titolo Markdown dopo; altrimenti restituisce il testo
    invariato.
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


# Il modello locale spesso sbaglia e confonde i nomi dei parametri dei tool.
# passando parametri sbagliati. La normalizzazione mappa i parametri giusti.
def normalize_tool_args(name: str, args: dict) -> dict:
    """
    Due famiglie di tool con parametro diverso:
      - tool di ricerca (web/RAG): vogliono 'query';
      - tool del Knowledge Graph: vogliono 'topic'.
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
                print(f"Argomento '{alias}' rimappato su '{target}' per '{name}'.")
                return fixed
        return args
    if name in query_tools:
        return _remap("query")
    if name in topic_tools:
        return _remap("topic")
    return args


# Analogamente alla logica di wants_post, stabiliamo se l'utente vuole che vengano aggiunti nuovi dati 
# oppure se vuole modificare la stesura del post in modo puramente testuale.
_MODIFICATION_NEEDS_RESEARCH = {
    "aggiungi", "aggiungere", "confronta", "confronto", "paragona", "paragone",
    "cerca", "ricerca", "trova", "includi", "includere", "dati su", "specifiche",
    "informazioni su", "approfondisci", "approfondire", "inserisci un confronto",
    "verifica", "fonti", "fonte", "prezzo", "prezzi", "scheda tecnica",
    "metti a confronto", "compara",
}

#Metodo che implementa quanto detto sopra. Determina se la modifica richiede una ricerca di nuovi dati o meno.
def modification_needs_research(feedback: str) -> bool:
    """
    True se la richiesta di modifica dell'utente sembra richiedere nuovi dati
    (e quindi un altro giro di ricerca), False se è una modifica solo testuale
    (accorcia, cambia tono, riscrivi l'introduzione, ecc.).
    """
    if not feedback:
        return False
    low = feedback.lower()
    return any(kw in low for kw in _MODIFICATION_NEEDS_RESEARCH)


# Analogamente a sopra, questi dizionari servono per la parte relativa allo scoping
# Il primo è vago solo se non c'è un tema principale, il secondo invece è sempre vago.
_VAGUE_STANDALONE = (
    "scrivimi qualcosa", "scrivi qualcosa", "scrivi un post", "scrivimi un post",
    "scrivi un articolo", "scrivimi un articolo", "dammi un articolo", "dammi un post",
    "parlami di auto", "parlami di moto", "parlami di motori",
    "parlami di una moto", "parlami di una auto", "parlami di un auto",
    "parlami di un moto", "parlami di una macchina", "parlami di un veicolo",
    "scrivi di una moto", "scrivi di una auto", "scrivi di un auto",
    "un post a caso", "un argomento qualsiasi", "scrivi un post su una moto",
    "scrivi un post su un auto", "scrivi un post su una macchina"
)
_VAGUE_ALWAYS = (
    "fai tu", "fai te", "decidi tu", "scrivi tu", "quello che vuoi",
    "qualcosa di interessante", "sorprendimi", "come preferisci", "scegli tu",
    "fai quello che vuoi", "scrivi quello che vuoi", "scrivi su qualcosa a caso"
)


def is_clearly_vague(user_input: str) -> bool:
    """
    True se la richiesta è palesemente generica. Serve
    per lo scoping: questi casi devono sempre far scattare il chiarimento, senza
    dipendere dal modello 3B.
    Se scrivo "Scrivi un post" deve risultare ovviamente vago, mentre se scrivo
    "Scrivi un post sulla nuova Ferrari" è ovvio che non lo sia." 
    """

    #Se non do input è totalmente vago 
    if not user_input or not user_input.strip():
        return True 

    low = " ".join(user_input.lower().replace("'", " ").split())

    # Caso sempre vago
    for pat in _VAGUE_ALWAYS:
        if pat in low:
            return True

    # Caso normale, con parole di "distanza".
    _filler = {"su", "di", "del", "della", "dei", "delle", "sul", "sulla", "sui",
               "sulle", "per", "un", "uno", "una", "il", "la", "lo", "le", "gli",
               "che", "a", "e", "con", "tema", "argomento", "qualcosa"}
    for pat in _VAGUE_STANDALONE:
        idx = low.find(pat)
        if idx != -1:
            # Vediamo cosa resta dopo l'applicazione di questa regola, 
            # se non resta nulla allora è ancora vago
            rest = low[idx + len(pat):].strip()
            rest_words = [w for w in rest.split() if w not in _filler]
            if not rest_words:
                return True

    # Richieste estremamente corte e senza sostanza
    generic_words = {"scrivi", "scrivimi", "post", "articolo", "auto", "moto",
                     "motori", "qualcosa", "un", "uno", "una", "di", "su", "il", "la"}
    words = low.split()
    if len(words) <= 3 and all(w in generic_words for w in words):
        return True
    return False


# Rimuoviamo articoli, preposizioni, parole editoriali e di taglio, in modo che
# un titolo del genere : "La Giulia Quadrifoglio: storia, simbolo e eredita' dinastica" e
# "Scrivi un post sulla Alfa Romeo Giulia Quadrifoglio" convergano su una chiave simile.
# Il modellino infatti tende sempre ad arricchire il titolo dei post.
_TOPIC_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "del", "dello",
    "della", "dei", "degli", "delle", "a", "ad", "da", "in", "con", "su", "sul",
    "sulla", "sui", "sulle", "per", "tra", "fra", "e", "ed", "o", "come", "che",
    "scrivi", "scrivimi", "parlami", "dammi", "post", "articolo", "un", "guida",
    "completa", "dettagliata", "recensione", "confronto", "storia", "simbolo",
    "eredita", "eredità", "dinastica", "significato", "descrizione", "tecnica",
    "approfondimento", "tutto", "sul", "vs", "chiarimento", "informazioni", "generali",
    "un", "uno", "una", "un auto", "una auto", "un auto elettrica", "una auto elettrica", "auto elettrica", 
    "moto", "motori", "motore", "motore elettrico", "motore a scoppio", "auto sportive",
    "auto sportive", 
}


# Metodo che normalizza il topic del post in modo da renderlo confrontabile con gli altri topic presenti nel KG.
# Viene usato dall'updater.py per decidere se creare un nuovo topic o aggiornarne uno esistente.
def canonical_topic(raw: str) -> str:
    """
    Genero una chiave normalizzata per il Knowledge Graph, in modo che i soggetti uguali
    convergano nella stessa chiave (e la gap-analysis capisca se quel soggetto è già stato trattato o no)

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
