
""" File di configurazione dell'intero agente, 
    dove scegliamo parametri operativi con la possibilità di sovrascriverli direttamente dell'env.
"""

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

# Usare il token HF elimina il warning "unauthenticated
# requests to the HF Hub" e abilita rate limit/download migliori.
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)
    os.environ.setdefault("HF_TOKEN", _hf_token)


@dataclass(kw_only=True)
class Configuration:
    """
    Configurazioni per l'agente AutomotiveBloggerAgent.
    I valori possono essere sovrascritti da variabili d'ambiente o da un RunnableConfig.
    """

    # Parametri dei modelli locali con Ollama
    # Ho scelto lo stesso modello tutto il progetto, e per la ricerca.
    model_name: str = "ministral-3:3b"
    draft_model_name: str = "ministral-3:3b"
    model_provider: str = "ollama"

    # Temperatura per i nodi deterministici in modo da avere un'output stabile e ripetibile
    temperature: float = 0.0

    # Temperatura per la stesura dell'articolo più alta di quella sopra per avere un output
    # più creativo
    draft_temperature: float = 0.6

    # Contesto del modello principale molto ampio (ai limiti della VRAM) per permettere al modello
    # di avere un buon ragioamento ReAct con tanti step e tool da utilizzare
    model_num_ctx: int = 22528

    # Contesto per la stesura più contenuto, non serve andare ai limiti della VRAM per la stesura.
    draft_num_ctx: int = 22528

    # Indice di sicurezza per il numero di token durante la stesura.
    # il modello potrebbe entrare in un loop ripetitivo e generare
    # un testo infinito, pertanto mettiamo un limite a 8K di token, quindi circa 4/5000 parole.
    draft_num_predict: int = int(os.environ.get("DRAFT_NUM_PREDICT", "8192"))

    # Soglie per il RAG, ne ho utilizzate due per cercare di avere un buon compromesso tra
    # accuratezza e pertinenza dei risultati. La prima infatti serve ad evitare che 
    # documenti non pertineti vengano inseriti nel post. Mentre la seconda serve per evitare che
    # nel caso in cui venissero scartati chunk "buoni", alzo la soglia ad 1.20, ma la tengo comunque
    # "bassa" per evitare di inserire falsi positiv.
    # top_k indica invece il numero massimo di chunk da reperire.
    rag_distance_threshold: float = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "1.10"))
    rag_fallback_max: float = float(os.environ.get("RAG_FALLBACK_MAX", "1.17"))
    rag_top_k: int = int(os.environ.get("RAG_TOP_K", "5"))

    # Modello utilizzato per riassumere i risultati delle ricerche tramite Tavily
    # invece che passare i risultati direttamente come fonte per la stesura, questi
    # vengono riassunti dallo stesso modello e poi messi a disposizione di se stesso.
    # La temperatura è bassa, i riassunti devono essere fedeli e il contesto più
    # alto possibile per non perdere informazioni essenziali.
    summarizer_model_name: str = "ministral-3:3b"
    summarizer_temperature: float = 0.2
    summarizer_num_ctx: int = 22528

    # Questi campi di "osservabilità" servono a capire se la configurazione contiene
    # le variabili necessarie per il tracing. Non attiva il tracing vero.
    
    langsmith_tracing: str = os.environ.get("LANGSMITH_TRACING", "false")
    langsmith_project: Optional[str] = os.environ.get("LANGSMITH_PROJECT", "AutomotiveBloggerAgent")

    # Valore booleano per attivare le diagnostiche (come tool call, RAG, ecc)
    # Per default e' disattivato, cosi' l'output e' pulito.
    
    debug: bool = os.environ.get("DEBUG", "false").strip().lower() == "true"


    # Funzione di utilita' per creare un'istanza di Configuration da un RunnableConfig

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Crea un'istanza di Configuration da un RunnableConfig"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        # converto "debug" da stringa a bool sennò da errore
        if isinstance(values.get("debug"), str):
            values["debug"] = values["debug"].strip().lower() == "true"
        return cls(**{k: v for k, v in values.items() if v is not None})




# Funzione di utilita' per controllare se LangSmith e' configurato correttamente
# nel main.py, un semplice "health-check" per capire se il tracing funziona o manca qualcosa.

def check_langsmith_setup() -> bool:
    tracing = os.environ.get("LANGSMITH_TRACING", "false").strip().lower() == "true"
    api_key = os.environ.get("LANGSMITH_API_KEY")
    project = os.environ.get("LANGSMITH_PROJECT", "(default)")

    if not tracing:
        print("Tracing disattivato. Le esecuzioni non verranno tracciate.")
        return False
    if not api_key:
        print("Tracing attivo ma manca LANGSMITH_API_KEY nel .env. Il tracing fallira'.")
        return False

    print(f"Tracing attivo. Progetto: '{project}'.")
    return True
