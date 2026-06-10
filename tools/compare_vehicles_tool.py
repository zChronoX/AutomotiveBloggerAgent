"""
Tool per la comparazione di veicoli (auto e moto)
Usa Tavily per prendere informazioni su i veicoli scelti e le confronta.
Usa due modelli: uno per la ricerca (ministral-3:3b) e uno per la comparazione (llama3.2:1b_fine_tuned).
"""


import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from prompts.tool_prompts import TINY_JUDGE_SYSTEM_PROMPT


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


RESEARCHER_MODEL = "ministral-3:3b"             # Per la sintesi e normalizzazione dei dati
JUDGE_MODEL = "llama3.2:1b_fine_tuned"          # Giudice per la comparazione dei modelli


# --- Prompt della pipeline di ricerca (definiti qui per tenere il tool auto-contenuto) ---
# Sostituiscono il vecchio VEHICLE_RESEARCH_PROMPT con una pipeline in DUE passi:
# 1) ELABORATE_SOURCES_PROMPT: rende le fonti leggibili SENZA comprimerle troppo
#    (il riassunto aggressivo perdeva i dati; il testo grezzo confondeva il modello).
# 2) SPEC_PROFILE_PROMPT: estrae il profilo nel FORMATO FISSO su cui e' stato
#    addestrato il modello fine-tuned (paragrafo di soli fatti, campi in ordine fisso).

ELABORATE_SOURCES_PROMPT = """Sei un analista dati del settore automotive. Qui sotto trovi il
testo (gia' ripulito) di {n_fonti} fonti web sul veicolo {veicolo}.

Riscrivi OGNI fonte in un testo scorrevole e leggibile, conservando TUTTI i dati tecnici
presenti: motore (tipo, cilindrata, cilindri), potenza (CV) e coppia (Nm), accelerazione,
consumi, prezzo, sicurezza (stelle Euro NCAP, ADAS), cambio/trazione, dotazione.
REGOLE:
- NON inventare nulla: riporta solo cio' che c'e' nel testo. NON aggiungere dati da memoria.
- NON comprimere troppo: massimo 10 frasi per fonte, ma tieni TUTTI i numeri.
- Mantieni le unita' di misura ESATTAMENTE come nell'originale.
- Salta navigazione, gallerie, opinioni dei lettori e parti non sul veicolo.
- Formato output: "FONTE 1: <testo>" e a capo "FONTE 2: <testo>" (se presente).

{testo_fonti}"""

SPEC_PROFILE_PROMPT = """Sei un analista dati del settore automotive.
Dalle fonti elaborate qui sotto, scrivi il profilo tecnico del veicolo {veicolo}
nel FORMATO FISSO richiesto.

FORMATO FISSO (un solo paragrafo, campi in QUEST'ORDINE, ometti i campi senza dato):
"Motore <tipo/cilindrata/cilindri> da <CV> CV e <Nm> Nm. 0-100 in <s> s.
Consumo <valore con unita'>. Prezzo da <EUR>. <N> stelle Euro NCAP, <ADAS principali>.
<Cambio/trazione e 1-2 voci di dotazione chiave>."

Esempio del risultato atteso:
"Motore 2.0 4 cilindri turbo da 421 CV e 500 Nm. 0-100 in 3.9 s. Consumo medio dichiarato
8.3 l/100km. Prezzo da 63.000 EUR. 5 stelle Euro NCAP, frenata automatica di serie.
Cambio doppia frizione 8 rapporti, trazione integrale, sedili sportivi."

REGOLE RIGIDE:
1. UN SOLO paragrafo discorsivo, MASSIMO 80 parole. Niente titoli, elenchi o Markdown.
2. SOLO numeri e fatti presenti nelle fonti elaborate. Se un dato manca, OMETTILO
   (NON scrivere "non disponibile", NON inventare, NON usare la memoria).
3. Niente aggettivi enfatici, niente note o sezioni aggiuntive.

FONTI ELABORATE:
{fonti_elaborate}

Profilo tecnico (un paragrafo, max 80 parole):"""


# Uso uno schema piatto invece che un JSON per evitare
# che il modello piccolo sbagli nel riempimento dei campi.

from typing import Optional


class CompareVehiclesInput(BaseModel):
    tipo: str = Field(description="Specifica se e' 'Auto' o 'Moto'")
    v1_marca: str = Field(description="Marca del primo veicolo (es. Fiat)")
    v1_modello: str = Field(description="Modello del primo veicolo (es. Panda)")
    v1_anno: Optional[str] = Field(default="", description="Anno del primo veicolo (es. 2024). Opzionale.")
    v1_motore: Optional[str] = Field(default="", description="Motorizzazione del primo veicolo (es. 1.0 Hybrid). Opzionale.")

    v2_marca: str = Field(description="Marca del secondo veicolo (es. Dacia)")
    v2_modello: str = Field(description="Modello del secondo veicolo (es. Sandero)")
    v2_anno: Optional[str] = Field(default="", description="Anno del secondo veicolo (es. 2024). Opzionale.")
    v2_motore: Optional[str] = Field(default="", description="Motorizzazione del secondo veicolo (es. 1.0 TCe). Opzionale.")

# Tiene insieme i dati del veicolo in modo pulito e ordinato
class VehicleSpec:
    def __init__(self, tipo, marca, modello, anno, motorizzazione):
        self.tipo = tipo
        self.marca = marca
        self.modello = modello
        self.anno = anno
        self.motorizzazione = motorizzazione


# --- Pulizia delle fonti web -----------------------------------------------------------
# Questo tool fa una ricerca Tavily PROPRIA e separata da quella del server MCP, quindi
# replichiamo qui la stessa logica di pulizia del server (dedup + filtro articoli validi +
# rimozione del rumore di pagina). Era proprio il testo grezzo sporco a far sbagliare/
# inventare le specifiche al modellino.

# Marcatori di righe di navigazione / box "correlati" / gallerie da scartare.
_NAV_MARKERS = (
    "fotogallery", "le ultime da", "di tendenza", "ultimi articoli", "consigliati per te",
    "leggi anche", "potrebbe interessarti", "guarda anche", "navigate su",
    "vogliamo la tua opinione", "il bar del", "newsletter", "iscriviti", "condividi",
    "seguici", "cookie",
)


def _dedup_results(results: list) -> list:
    """Toglie i risultati duplicati per URL (Tavily a volte li ripete)."""
    seen, unique = set(), []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)
    return unique


def _is_valid_article(r: dict) -> bool:
    """Scarta sitemap, feed, pagine-indice e contenuti troppo poveri (stessa logica del
    server MCP): cosi' al ricercatore arrivano solo articoli veri."""
    url = (r.get("url") or "").lower()
    title = (r.get("title") or "").lower()
    content = r.get("content") or ""
    bad_markers = ("sitemap", ".xml", "/feed", "rss", "/tag/", "/category/", "/categoria/")
    if any(m in url for m in bad_markers):
        return False
    if "[xml]" in title or "sitemap" in title:
        return False
    if content.count("http") > 5:   # tante "http" = indice/lista di link, non un articolo
        return False
    if len(content.strip()) < 80:
        return False
    return True


def _clean_source_text(text: str) -> str:
    """Ripulisce il testo grezzo di una pagina: elimina righe duplicate (es. i titoli di
    gallerie ripetuti come 'Audi RS 3 Sportback (2024)'), le righe di navigazione/box
    correlati e le righe vuote, mantenendo prosa e dati."""
    out, seen = [], set()
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(m in low for m in _NAV_MARKERS):
            continue
        if low in seen:            # riga duplicata (gallerie/menu ripetuti)
            continue
        seen.add(low)
        out.append(line)
    return "\n".join(out).strip()


# Prima parte di ricerca dei dati
# arrichisco la query con i dati del veicolo più keywords specifiche
# la riceca va fatta solo su siti precisi.
def deep_research_vehicle(vehicle: VehicleSpec) -> str:
    """Ricerca mirata su fonti autorevoli e profilo tecnico nel formato del modello fine-tuned.
    Pipeline (replica la logica del search server MCP, come per la ricerca web "ufficiale"):
    ricerca topic 'general' (MAI news) sulla whitelist -> dedup -> filtro articoli validi ->
    pulizia del testo -> 2 fonti al massimo -> elaborazione LEGGIBILE che conserva i dati ->
    estrazione del profilo nel FORMATO FISSO atteso dal modello fine-tuned."""
    query = (f"{vehicle.tipo} {vehicle.marca} {vehicle.modello} {vehicle.anno} "
             f"{vehicle.motorizzazione} scheda tecnica prova su strada consumi prezzo")
    nome_veicolo = f"{vehicle.marca} {vehicle.modello} {vehicle.anno}".strip()

    # Whitelist di sole testate automotive autorevoli (niente social/video).
    siti_autorevoli = [
        "quattroruote.it", "alvolante.it", "motor1.com", "automoto.it", "omniauto.it",
        "hdmotori.it", "motorbox.com", "moto.it", "dueruote.it", "insella.it",
    ]

    # Ricerca primaria: topic SEMPRE 'general' (le news riportano lanci/indiscrezioni, non
    # schede tecniche), advanced, con testo integrale. Prendo piu' candidati e poi filtro.
    results = []
    try:
        risposta = tavily_client.search(
            query=query, search_depth="advanced", max_results=4, topic="general",
            include_domains=siti_autorevoli, include_raw_content=True,
        )
        results = risposta.get("results", []) if isinstance(risposta, dict) else []
    except Exception as e:
        print(f"[Avviso] Errore Tavily (ricerca avanzata) per {vehicle.marca}: {e}")

    # Fallback: ricerca 'basic' MA SEMPRE sulla whitelist e topic 'general'. NON sul web
    # aperto: era da li' che entravano YouTube/Facebook con dati inattendibili.
    if not results:
        try:
            risposta = tavily_client.search(
                query=query, search_depth="basic", max_results=4, topic="general",
                include_domains=siti_autorevoli, include_raw_content=True,
            )
            results = risposta.get("results", []) if isinstance(risposta, dict) else []
        except Exception:
            results = []

    # Dedup + filtro articoli validi + pulizia del testo; tengo al massimo 2 fonti.
    results = [r for r in _dedup_results(results) if _is_valid_article(r)]
    blocchi = []
    for r in results[:2]:
        grezzo = (r.get("raw_content") or r.get("content") or "")
        clean = _clean_source_text(grezzo)
        if clean:
            blocchi.append(clean[:3000])   # cap per fonte: dati a sufficienza, niente muri di testo

    if not blocchi:
        return f"Dati web non disponibili per {nome_veicolo}."

    researcher = ChatOllama(model=RESEARCHER_MODEL, temperature=0.0, keep_alive="2m")
    # keep_alive="2m" (e non 0): qui facciamo DUE chiamate consecutive allo stesso modello,
    # scaricarlo e ricaricarlo tra una e l'altra raddoppierebbe la latenza. E' lo stesso
    # modello del "cervello" dell'agente, quindi non occupa VRAM aggiuntiva.

    # Passo 1: elaborazione LEGGIBILE delle fonti (conserva tutti i dati, niente compressione
    # aggressiva): e' il testo "comprensibile" su cui lavora l'estrazione.
    testo_fonti = "\n\n--- FONTE SUCCESSIVA ---\n\n".join(blocchi)
    prompt_elab = ELABORATE_SOURCES_PROMPT.format(
        n_fonti=len(blocchi), veicolo=nome_veicolo, testo_fonti=testo_fonti
    )
    try:
        fonti_elaborate = (researcher.invoke([HumanMessage(content=prompt_elab)]).content or "").strip()
    except Exception:
        fonti_elaborate = testo_fonti  # in caso di errore uso il testo pulito grezzo

    # Passo 2: profilo tecnico nel FORMATO FISSO atteso dal modello fine-tuned.
    prompt_profile = SPEC_PROFILE_PROMPT.format(veicolo=nome_veicolo, fonti_elaborate=fonti_elaborate)
    summary = researcher.invoke([HumanMessage(content=prompt_profile)])
    profile = (summary.content or "").strip()

    # Il modellino fine tuned è stato addestrato con profili lunghi 150-200 token (max 2048).
    # Per sicurezza tronco il profilo a 900 caratteri. Sennò genererà un'output instabile.
    MAX_PROFILE_CHARS = 900
    if len(profile) > MAX_PROFILE_CHARS:
        profile = profile[:MAX_PROFILE_CHARS].rsplit(" ", 1)[0] + "..."
    return profile


# Tool effettivo di comparazione.
# Chiamo parallelamente due ricerche con Ministral.

@tool("compare_vehicles", args_schema=CompareVehiclesInput)
def compare_vehicles_tool(tipo: str, v1_marca: str, v1_modello: str,
                          v2_marca: str, v2_modello: str,
                          v1_anno: str = "", v1_motore: str = "",
                          v2_anno: str = "", v2_motore: str = "") -> str:
    """
    Usa questo tool SOLO per confrontare due auto o moto. Indica marca e modello di entrambi
    (anno e motorizzazione sono opzionali: se non li conosci, lasciali vuoti).
    """
    veicolo_1 = VehicleSpec(tipo, v1_marca, v1_modello, v1_anno, v1_motore)
    veicolo_2 = VehicleSpec(tipo, v2_marca, v2_modello, v2_anno, v2_motore)

    print(f"\nComparazione di {veicolo_1.marca} {veicolo_1.modello} "
          f"vs {veicolo_2.marca} {veicolo_2.modello}")

    dati_v1 = deep_research_vehicle(veicolo_1)
    dati_v2 = deep_research_vehicle(veicolo_2)

    judge = ChatOllama(model=JUDGE_MODEL, temperature=0.1, keep_alive=0)
    system_prompt = SystemMessage(content=TINY_JUDGE_SYSTEM_PROMPT)
    user_prompt = HumanMessage(content=(
        f"<veicolo_1>\nMarca: {veicolo_1.marca}\nModello: {veicolo_1.modello}\n"
        f"Dati estratti: {dati_v1}\n</veicolo_1>\n\n"
        f"<veicolo_2>\nMarca: {veicolo_2.marca}\nModello: {veicolo_2.modello}\n"
        f"Dati estratti: {dati_v2}\n</veicolo_2>\n\n"
        "Genera la comparazione seguendo le istruzioni del sistema."
    ))

    print("Elaborazione del verdetto in corso")
    verdetto = judge.invoke([system_prompt, user_prompt])
    return verdetto.content