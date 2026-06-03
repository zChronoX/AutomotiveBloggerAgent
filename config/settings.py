import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

# Se nel .env e' presente HF_TOKEN, lo esponiamo come variabile d'ambiente che le
# librerie HuggingFace leggono automaticamente: elimina il warning "unauthenticated
# requests to the HF Hub" e abilita rate limit/download migliori. Va fatto QUI perche'
# configuration.py e' importato prima del caricamento dei modelli di embedding.
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)
    os.environ.setdefault("HF_TOKEN", _hf_token)


@dataclass(kw_only=True)
class Configuration:
    """
    Configurazioni per l'agente Blogger Copilot.

    La struttura (dataclass kw_only + from_runnable_config) ricalca il
    configuration.py del tutorial "agents-from-scratch": i valori possono essere
    sovrascritti da variabili d'ambiente (in MAIUSCOLO) o da un RunnableConfig.
    """

    # --- Parametri dei modelli locali (Ollama) - ARCHITETTURA IBRIDA ---
    # Ogni modello fa cio' in cui eccelle:
    #  - model_name (Granite): il "cervello" - planning, scelta dei tool, ricerca, grading,
    #    routing, update KG. Granite 4.1 3B sceglie i tool in modo affidabile ed e' veloce.
    #  - draft_model_name (Ministral 8B): la "penna" - SOLO la stesura dell'articolo finale.
    #    Ministral produce prosa piu' ricca e discorsiva, adatta a un blog. La sua lentezza
    #    pesa una volta sola (il drafting), non sull'intero flusso ReAct.
    model_name: str = "ministral-3:3b"
    draft_model_name: str = "ministral-3:3b"
    model_provider: str = "ollama"

    # Temperatura per i nodi DETERMINISTICI (planning, grading, routing, update KG):
    # 0.0 = output stabile e ripetibile, ideale dove serve precisione e non creativita'.
    temperature: float = 0.0

    # Temperatura per la SOLA stesura dell'articolo (drafting_node):
    # piu' alta per ottenere prosa meno piatta e meno ripetitiva.
    # Usata in blogger_agent.py da un client LLM dedicato alla stesura.
    draft_temperature: float = 0.6

    # Finestra di contesto dei modelli: se non impostata, Ollama usa un default piccolo
    # (2048 token) che puo' TRONCARE prompt+cronologia quando ci sono molti tool da esporre
    # e una conversazione ReAct lunga -> il modello "vede" meno e sceglie peggio i tool.
    # Granite supporta contesti ampi: 32768 da' margine abbondante. Verificato sperimentalmente
    # che un contesto adeguato migliora nettamente la scelta dei tool.
    model_num_ctx: int = 16384

    # Contesto SEPARATO per il modello di stesura (Ministral 8B). La stesura riceve un input
    # contenuto (topic + fonti + linee guida), non l'intera cronologia ReAct con tutti i tool,
    # quindi NON serve un contesto enorme. Inoltre un 8B con KV cache da 32K rischia di sforare
    # la VRAM (spillover in RAM = molto lento). 8192 e' ampiamente sufficiente per un articolo.
    draft_num_ctx: int = 16384

    # --- Modello RIASSUNTORE del server MCP di ricerca (mcp_search_server.py) ---
    # E' un modello SEPARATO dal modello del grafo: legge molte pagine web (Tavily)
    # e ne produce un riassunto tecnico denso. Usa una context ampia (num_ctx) per
    # leggere piu' contenuto possibile; temperatura bassa per un riassunto fedele.
    # NOTA: su GPU con poca VRAM (es. 8GB) num_ctx alto puo' causare spillover in RAM
    # (piu' lento) o OOM: e' un valore SPERIMENTALE da tarare sulla propria macchina.
    summarizer_model_name: str = "phi4-mini:latest"
    summarizer_temperature: float = 0.2
    summarizer_num_ctx: int = 22528

    # --- Osservabilita' LangSmith (requisito di valutazione del PDF) ---
    # NOTA IMPORTANTE: questi campi servono solo a DOCUMENTARE/ISPEZIONARE la
    # configurazione. Il tracing di LangSmith NON si attiva da qui: si attiva
    # tramite le variabili d'ambiente lette da LangChain all'import
    # (LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT nel file .env).
    # Qui le rileggiamo solo per poterle validare e mostrare a colpo d'occhio.
    langsmith_tracing: str = os.environ.get("LANGSMITH_TRACING", "false")
    langsmith_project: Optional[str] = os.environ.get("LANGSMITH_PROJECT", "blogger-copilot")

    # --- Modalita' DEBUG ---
    # Quando attiva (DEBUG=true nel .env), stampa le diagnostiche di sviluppo
    # ([DIAG] tool_call, [DIAG RAG] distanze, ecc.). Di default e' DISATTIVATA,
    # cosi' l'output e' pulito per l'uso normale e per la demo. Le diagnostiche
    # restano nel codice (utili per ispezionare il comportamento), ma silenziose.
    debug: bool = os.environ.get("DEBUG", "false").strip().lower() == "true"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Crea un'istanza di Configuration da un RunnableConfig (pattern del tutorial)."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        # 'debug' arriva come stringa dall'ambiente ("true"/"false"): convertilo a bool,
        # altrimenti la stringa "false" risulterebbe truthy in Python.
        if isinstance(values.get("debug"), str):
            values["debug"] = values["debug"].strip().lower() == "true"
        # Filtra i valori None prima di inizializzare la classe
        return cls(**{k: v for k, v in values.items() if v is not None})


def check_langsmith_setup() -> bool:
    """
    Diagnostica leggibile dell'osservabilita' LangSmith.

    Da chiamare all'avvio (es. in main.py) per sapere SUBITO se le run verranno
    tracciate, invece di scoprirlo a fine progetto. Non attiva nulla: si limita a
    controllare le variabili d'ambiente che LangChain usa per il tracing.

    Restituisce True se il tracing risulta configurato correttamente.
    """
    tracing = os.environ.get("LANGSMITH_TRACING", "false").strip().lower() == "true"
    api_key = os.environ.get("LANGSMITH_API_KEY")
    project = os.environ.get("LANGSMITH_PROJECT", "(default)")

    if not tracing:
        print("[LangSmith] Tracing DISATTIVATO (LANGSMITH_TRACING != 'true'). "
              "Le esecuzioni non verranno tracciate.")
        return False
    if not api_key:
        print("[LangSmith] ATTENZIONE: LANGSMITH_TRACING=true ma manca LANGSMITH_API_KEY nel .env. "
              "Il tracing fallira'.")
        return False

    print(f"[LangSmith] Tracing ATTIVO. Progetto: '{project}'.")
    return True
