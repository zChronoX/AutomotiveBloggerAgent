"""
Tool per la comparazione di veicoli (auto e moto).
Per ogni veicolo recupera le fonti tramite lo STESSO tool di ricerca dell'agente
(mcp_web_search -> server MCP), poi le trasforma nel profilo a formato fisso e infine
le confronta. Usa due modelli: ministral-3:3b (per il profilo) e
llama3.2:1b_fine_tuned (giudice della comparazione).
"""


from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.tool_prompts import TINY_JUDGE_SYSTEM_PROMPT, SPEC_PROFILE_PROMPT


RESEARCHER_MODEL = "ministral-3:3b"             # Per la sintesi e normalizzazione dei dati
JUDGE_MODEL = "llama3.2:1b_fine_tuned"          # Giudice per la comparazione dei modelli


# --- Prompt usati da questo tool ---
# Definiti in prompts/tool_prompts.py (coerenza con gli altri prompt del progetto)
# e importati in cima a questo modulo:
#  - SPEC_PROFILE_PROMPT: trasforma le fonti restituite da mcp_web_search nel profilo a
#    FORMATO FISSO su cui e' stato addestrato il modello fine-tuned (paragrafo di soli fatti).
#  - TINY_JUDGE_SYSTEM_PROMPT: prompt di sistema del giudice fine-tuned (formato del verdetto).
# La sintesi delle singole fonti NON avviene qui: la fa il server MCP (mcp_web_search).


# Uso uno schema piatto invece che un JSON per evitare
# che il modello piccolo sbagli nel riempimento dei campi.

from typing import Optional


class CompareVehiclesInput(BaseModel):
    # Schema SEMPLICE (2 sole stringhe obbligatorie). Lo schema precedente aveva 5 campi
    # obbligatori separati (tipo, marca/modello x2) e il modello 3B ne perdeva sempre
    # qualcuno -> ValidationError e tool fallito (es. passava solo il primo veicolo).
    # Con due nomi completi il modello riempie l'input in modo affidabile.
    veicolo_1: str = Field(description="Nome completo del primo veicolo, es. 'Volkswagen Golf GTI'")
    veicolo_2: str = Field(description="Nome completo del secondo veicolo, es. 'Toyota GR Yaris'")
    tipo: Optional[str] = Field(default="", description="'Auto' o 'Moto'. Opzionale.")

# Tiene insieme i dati del veicolo in modo pulito e ordinato
class VehicleSpec:
    def __init__(self, tipo, marca, modello, anno, motorizzazione):
        self.tipo = tipo
        self.marca = marca
        self.modello = modello
        self.anno = anno
        self.motorizzazione = motorizzazione


# Cache a livello di modulo per evitare che il 3B richiami compare_vehicles 3 volte
# nello stesso loop ReAct (succede: il modello non si ricorda di averlo gia' fatto).
# La chiave e' la coppia (marca1_modello1, marca2_modello2) normalizzata.
_compare_cache: dict[str, str] = {}


# Prima parte di ricerca dei dati
# arrichisco la query con i dati del veicolo più keywords specifiche
# la riceca va fatta solo su siti precisi.
def deep_research_vehicle(vehicle: VehicleSpec) -> str:
    """Profilo tecnico del veicolo nel formato del modello fine-tuned.
    Pipeline (per OGNI veicolo, quindi 2 ricerche in totale per confronto):
    1) UNA ricerca con lo STESSO tool dell'agente (mcp_web_search): va in HTTP al server
       MCP, che esegue ricerca + riassunto per fonte NEL SUO processo e restituisce il
       testo gia' pulito (formato "FONTE 1/2/3").
    2) Dal risultato tengo solo le PRIME 3 FONTI.
    3) UNA chiamata al modello per trasformare quel formato nel profilo a campi fissi
       atteso dal fine-tuned.
    """
    # Query SEMPLICE e deterministica: marca + modello (+ anno se c'e').
    # Le query lunghe e "ricche" peggiorano i risultati di Tavily: bastano i termini chiave.
    nome_veicolo = f"{vehicle.marca} {vehicle.modello} {vehicle.anno}".strip()
    query = f"{vehicle.marca} {vehicle.modello} {vehicle.anno} scheda tecnica".strip()

    # 1) Ricerca via lo STESSO tool dell'agente: mcp_web_search. Va in HTTP al server MCP
    # (porta 8765), quindi la ricerca e i riassunti per-fonte girano NEL processo del
    # server (non qui in main.py) e al compare torna SOLO il testo gia' pulito, nello
    # stesso formato "FONTE 1/2/3" della ricerca normale. Import lazy per non eseguire
    # il modulo del client al caricamento dei tool.
    try:
        from tools.mcp_client_tool import mcp_web_search
        # mcp_web_search e' un @tool LangChain: lo invochiamo con .invoke (o .func se serve).
        try:
            testo_ricerca = mcp_web_search.invoke({"query": query}) or ""
        except Exception:
            testo_ricerca = mcp_web_search.func(query) or ""
    except Exception as e:
        print(f"[Avviso] Ricerca web (MCP) non disponibile per {nome_veicolo}: {e}")
        testo_ricerca = ""

    if not testo_ricerca.strip() or testo_ricerca.lower().startswith("nessun"):
        return f"Dati web non disponibili per {nome_veicolo}."

    # 2) Tengo solo le PRIME 3 FONTI restituite dal server (formato "FONTE 1: ... FONTE 2: ...").
    # Il server cerca su 5 fonti; per non rallentare ancora di piu' il giudizio ne uso 3.
    quarta = testo_ricerca.find("FONTE 4:")
    if quarta != -1:
        testo_ricerca = testo_ricerca[:quarta].rstrip()

    # 3) UNA chiamata al modello: trasforma il formato della ricerca nel profilo fisso.
    researcher = ChatOllama(model=RESEARCHER_MODEL, temperature=0.0, keep_alive="2m")
    prompt_profile = SPEC_PROFILE_PROMPT.format(veicolo=nome_veicolo, fonti_elaborate=testo_ricerca)
    summary = researcher.invoke([HumanMessage(content=prompt_profile)])
    profile = (summary.content or "").strip()

    # Il modellino fine tuned è stato addestrato con profili lunghi 150-200 token (max 2048).
    # Per sicurezza tronco il profilo a 900 caratteri. Sennò genererà un'output instabile.
    MAX_PROFILE_CHARS = 900
    if len(profile) > MAX_PROFILE_CHARS:
        profile = profile[:MAX_PROFILE_CHARS].rsplit(" ", 1)[0] + "..."
    return profile

# Divide un nome completo in (marca, modello): la prima parola e' la marca,
# il resto il modello (es. "Volkswagen Golf GTI" -> "Volkswagen", "Golf GTI").
def _split_nome(nome: str) -> tuple:
    parti = (nome or "").strip().split()
    if not parti:
        return ("", "")
    if len(parti) == 1:
        return (parti[0], "")
    return (parti[0], " ".join(parti[1:]))


# Tool effettivo di comparazione fra due veicoli.
@tool("compare_vehicles", args_schema=CompareVehiclesInput)
def compare_vehicles_tool(veicolo_1: str, veicolo_2: str, tipo: str = "") -> str:
    """
    Usa questo tool SOLO per CONFRONTARE DUE veicoli (auto o moto), in un'unica chiamata
    con il NOME COMPLETO di entrambi (es. veicolo_1='Volkswagen Golf GTI',
    veicolo_2='Toyota GR Yaris').
    REGOLA TASSATIVA: serve SEMPRE due veicoli. NON usarlo MAI per cercare i dati di un
    solo modello: in quel caso usa 'fetch_vehicle_specs'.
    """
    m1, mod1 = _split_nome(veicolo_1)
    m2, mod2 = _split_nome(veicolo_2)
    v1 = VehicleSpec(tipo, m1, mod1, "", "")
    v2 = VehicleSpec(tipo, m2, mod2, "", "")

    # Cache: il 3B nel loop ReAct a volte richiama compare_vehicles 2-3 volte di fila
    # perche' non si ricorda di averlo gia' fatto. Restituisco il risultato gia' calcolato.
    cache_key = f"{veicolo_1}_vs_{veicolo_2}".lower().strip()
    if cache_key in _compare_cache:
        print(f"\nComparazione di {veicolo_1} vs {veicolo_2} (risultato gia' in cache)")
        return _compare_cache[cache_key]

    print(f"\nComparazione di {veicolo_1} vs {veicolo_2}")

    dati_v1 = deep_research_vehicle(v1)
    dati_v2 = deep_research_vehicle(v2)

    # Giudice fine-tuned: chiamata IDENTICA all'originale (nessun num_predict, che cappava
    # l'output e tagliava il verdetto). keep_alive=0: scarica il modello dopo l'uso.
    judge = ChatOllama(model=JUDGE_MODEL, temperature=0.1, keep_alive=0)
    system_prompt = SystemMessage(content=TINY_JUDGE_SYSTEM_PROMPT)
    user_prompt = HumanMessage(content=(
        f"<veicolo_1>\nMarca: {v1.marca}\nModello: {v1.modello}\n"
        f"Dati estratti: {dati_v1}\n</veicolo_1>\n\n"
        f"<veicolo_2>\nMarca: {v2.marca}\nModello: {v2.modello}\n"
        f"Dati estratti: {dati_v2}\n</veicolo_2>\n\n"
        "Genera la comparazione seguendo le istruzioni del sistema."
    ))

    print("Elaborazione del verdetto in corso")
    verdetto = judge.invoke([system_prompt, user_prompt])
    result = (verdetto.content or "").strip()

    # Il modellino 1B a volte si ferma dopo le 4 categorie e OMETTE il "Verdetto Finale".
    # Se manca, lo chiedo con UNA chiamata mirata e lo appendo: cosi' il verdetto c'e'
    # sempre, senza inventare nulla (si basa sulla comparazione appena prodotta).
    if "verdetto" not in result.lower():
        try:
            chiusura = judge.invoke([
                SystemMessage(content=(
                    "Sei un esperto di comparazioni tra veicoli. Ti viene data una "
                    "comparazione per categorie tra due veicoli. Scrivi SOLO il "
                    "'Verdetto Finale': un unico paragrafo di 2-3 frasi che dice a chi "
                    "conviene il primo veicolo e a chi il secondo. Niente altro.")),
                HumanMessage(content=(
                    f"Veicoli: {veicolo_1} vs {veicolo_2}.\n\n"
                    f"Comparazione per categorie:\n{result}\n\nVerdetto Finale:")),
            ])
            testo_chiusura = (chiusura.content or "").strip()
            if testo_chiusura:
                result += f"\n\n**Verdetto Finale**\n{testo_chiusura}"
        except Exception:
            pass

    # Salvo in cache cosi' se il 3B richiama il tool, non ripeto la ricerca.
    _compare_cache[cache_key] = result
    return result