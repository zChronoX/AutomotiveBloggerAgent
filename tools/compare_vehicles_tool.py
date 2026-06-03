import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from prompts.tool_prompts import PHI4_VEHICLE_RESEARCH_PROMPT, TINY_JUDGE_SYSTEM_PROMPT

# Client Tavily
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Nomi dei modelli (allineati a `ollama list`)
RESEARCHER_MODEL = "phi4-mini:latest"          # sintesi fattuale dei dati grezzi
JUDGE_MODEL = "llama3.2:1b_fine_tuned"          # giudice fine-tuned per la comparazione


# ==========================================
# SCHEMI DI INPUT (APPIATTITI PER MINISTRAL)
# Schema "piatto" (campi semplici invece di oggetti annidati): i modelli locali
# popolano gli argomenti del tool in modo molto piu' affidabile cosi'.
# ==========================================
class CompareVehiclesInput(BaseModel):
    tipo: str = Field(description="Specifica se e' 'Auto' o 'Moto'")
    v1_marca: str = Field(description="Marca del primo veicolo (es. Fiat)")
    v1_modello: str = Field(description="Modello del primo veicolo (es. Panda)")
    v1_anno: str = Field(description="Anno del primo veicolo (es. 2024)")
    v1_motore: str = Field(description="Motorizzazione del primo veicolo (es. 1.0 Hybrid)")

    v2_marca: str = Field(description="Marca del secondo veicolo (es. Dacia)")
    v2_modello: str = Field(description="Modello del secondo veicolo (es. Sandero)")
    v2_anno: str = Field(description="Anno del secondo veicolo (es. 2024)")
    v2_motore: str = Field(description="Motorizzazione del secondo veicolo (es. 1.0 TCe)")


class VehicleSpec:
    """Raggruppa i dati di un veicolo internamente al tool."""
    def __init__(self, tipo, marca, modello, anno, motorizzazione):
        self.tipo = tipo
        self.marca = marca
        self.modello = modello
        self.anno = anno
        self.motorizzazione = motorizzazione


# ==========================================
# RICERCA + SINTESI FATTUALE
# ==========================================
def deep_research_vehicle(vehicle: VehicleSpec) -> str:
    """Ricerca mirata su fonti autorevoli e sintesi tecnica fattuale con il modello ricercatore."""
    query = (f"{vehicle.tipo} {vehicle.marca} {vehicle.modello} {vehicle.anno} "
             f"{vehicle.motorizzazione} prova su strada recensione consumi reali")

    siti_autorevoli = [
        "quattroruote.it", "alvolante.it", "motor1.com", "gazzetta.it/motori",
        "moto.it", "dueruote.it", "insella.it",
    ]

    testo_grezzo = ""
    # Tentativo 1: ricerca avanzata ristretta a domini autorevoli
    try:
        risposta = tavily_client.search(
            query=query, search_depth="advanced", max_results=2, include_domains=siti_autorevoli
        )
        for res in risposta.get("results", []):
            testo_grezzo += res.get("content", "") + "\n\n"
    except Exception as e:
        print(f"[Avviso] Errore Tavily (ricerca avanzata) per {vehicle.marca}: {e}")

    # Tentativo 2 (fallback): ricerca base senza restrizione di dominio
    if not testo_grezzo.strip():
        try:
            risposta = tavily_client.search(query=query, search_depth="basic", max_results=2)
            for res in risposta.get("results", []):
                testo_grezzo += res.get("content", "") + "\n\n"
        except Exception:
            testo_grezzo = "Dati web non disponibili."

    researcher = ChatOllama(model=RESEARCHER_MODEL, temperature=0.0, keep_alive=0)
    prompt = PHI4_VEHICLE_RESEARCH_PROMPT.format(query_base=query, testo_grezzo=testo_grezzo)
    summary = researcher.invoke([HumanMessage(content=prompt)])
    return summary.content


# ==========================================
# TOOL DI COMPARAZIONE
# ==========================================
@tool("compare_vehicles", args_schema=CompareVehiclesInput)
def compare_vehicles_tool(tipo: str, v1_marca: str, v1_modello: str, v1_anno: str, v1_motore: str,
                          v2_marca: str, v2_modello: str, v2_anno: str, v2_motore: str) -> str:
    """
    Usa questo tool SOLO per confrontare due auto o moto. Inserisci i dati precisi per entrambi i veicoli.
    """
    veicolo_1 = VehicleSpec(tipo, v1_marca, v1_modello, v1_anno, v1_motore)
    veicolo_2 = VehicleSpec(tipo, v2_marca, v2_modello, v2_anno, v2_motore)

    print(f"\n[Tool Comparatore] Ricerca profonda: {veicolo_1.marca} {veicolo_1.modello} "
          f"vs {veicolo_2.marca} {veicolo_2.modello}...")

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

    print("[Tool Comparatore] Elaborazione del verdetto (modello giudice)...")
    verdetto = judge.invoke([system_prompt, user_prompt])
    return verdetto.content