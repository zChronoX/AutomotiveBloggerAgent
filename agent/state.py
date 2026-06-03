"""
Definizione dello stato dell'agente e schemi di pianificazione.
Contenuto originale da schemas.py, spostato qui.

Soddisfa i requisiti:
- MCP / State Management: stato esplicito aggiornato ad ogni step.
- Planning: schema strutturato per la sequenza di post.
"""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from typing import List, Optional
from langgraph.graph import MessagesState


# ============================================================
# SCHEMI DI SCOPING (requisito ripreso dal tutorial Deep Research, notebook 1)
# User Clarification + Brief Generation
# ============================================================
class ClarifyWithUser(BaseModel):
    """
    Decisione strutturata sulla necessita' di chiarimenti.
    Ripreso dal notebook 1_scoping.ipynb del tutorial Deep Research di LangGraph.
    """
    need_clarification: bool = Field(
        description="True se la richiesta dell'utente e' troppo vaga/ambigua e servono chiarimenti prima di procedere."
    )
    question: str = Field(
        description="La domanda di chiarimento da porre all'utente (vuota se non serve chiarimento)."
    )
    verification: str = Field(
        description="Breve conferma di cosa si e' capito della richiesta (usata quando non serve chiarimento)."
    )


class ResearchBrief(BaseModel):
    """
    Brief strutturato che trasforma la richiesta dell'utente in un obiettivo editoriale chiaro.
    Ripreso dal notebook 1_scoping.ipynb (ResearchQuestion), adattato al dominio blog automotive.
    """
    refined_topic: str = Field(
        description="Il tema del post riformulato in modo chiaro e specifico, pronto per la pianificazione."
    )
    angle: str = Field(
        description="L'angolo/taglio editoriale (es. recensione tecnica, guida pratica, confronto, novita')."
    )
    notes: str = Field(
        description="Note utili al planning: aspetti da coprire, vincoli, pubblico di riferimento."
    )


# ============================================================
# SCHEMI DI PIANIFICAZIONE (requisito "Planning")
# ============================================================
class PostPlan(BaseModel):
    """Pianificazione di un singolo post del blog."""
    topic: str = Field(description="L'argomento principale del post")
    justification: str = Field(
        description="Giustificazione editoriale per la scelta di questo argomento (perche' e perche' ORA)"
    )
    post_category: Literal["events", "how_to", "review", "news"] = Field(
        description="La categoria del post (es. eventi imminenti, tutorial, recensione, news)"
    )


class PlanningSchema(BaseModel):
    """Output del processo di pianificazione editoriale: una SEQUENZA di post."""
    reasoning: str = Field(
        description="Ragionamento passo-passo per selezione e ordine, con attenzione a diversita' e copertura"
    )
    planned_posts: List[PostPlan] = Field(description="Sequenza ordinata dei post pianificati")


# ============================================================
# INPUT / STATO (requisito "MCP / State Management")
# ============================================================
class StateInput(TypedDict):
    """Input iniziale fornito dall'utente per avviare l'agente."""
    user_input: str


class State(MessagesState):
    """
    Stato globale dell'agente = il "contesto" gestito esplicitamente.
    Eredita 'messages' da MessagesState (come nel tutorial), che funge anche da
    contenitore degli output dei tool.

    Mappa con i campi richiesti dalle specifiche (State Management):
      - user_input      -> input utente
      - messages        -> tool outputs (ereditato)
      - reasoning_trace -> traccia di ragionamento (ReAct)
      - kg_summary      -> sintesi del Knowledge Graph
      - planning_info   -> informazioni di pianificazione (sequenza di post)
    Tutti i campi vengono effettivamente popolati durante l'esecuzione e riusati
    come input dei passi successivi.
    """
    # --- Input ---
    user_input: str

    # --- Scoping (clarification + brief, dal notebook 1 Deep Research) ---
    # Brief editoriale strutturato generato dopo l'eventuale chiarimento.
    research_brief: Optional[str]
    # Conteggio dei giri di chiarimento (per evitare loop infiniti di domande).
    clarification_count: Optional[int]

    # --- Pianificazione / topic ---
    current_topic: Optional[str]
    # NB: lista di DICT (non di oggetti PostPlan): il checkpointer di LangGraph serializza
    # nativamente i dict, mentre gli oggetti Pydantic generano il warning
    # "Deserializing unregistered type ... PostPlan" (bloccante in versioni future).
    # I PostPlan vengono comunque generati dal planner con with_structured_output e poi
    # convertiti in dict via .model_dump() prima di entrare nello State.
    planning_info: Optional[List[dict]]

    # --- Knowledge Graph ---
    kg_summary: Optional[str]

    # --- Trend RSS ---
    trends_summary: Optional[str]

    # --- Ragionamento ReAct ---
    reasoning_trace: Optional[str]

    # --- Grounding / citazioni (K-RAG) ---
    sources: Optional[List[str]]

    # Documenti locali recuperati deterministicamente nel K-RAG (research_agent_node),
    # passati al drafting come fonti citabili.
    local_sources: Optional[List[str]]

    # Note grezze (osservazioni complete dei tool) raccolte durante la ricerca.
    # Ispirato ai 'raw_notes' del notebook 2 Deep Research: conserviamo l'output integrale
    # dei tool oltre al riassunto, utile in fase di drafting/modifica per recuperare
    # dettagli che il riassunto potrebbe aver tagliato.
    raw_notes: Optional[List[str]]

    # Flag del guardrail "verifica fonti": True dopo che il sistema ha forzato una
    # ricerca web perche' il modello stava per scrivere senza alcuna fonte. Evita loop.
    forced_web_search: Optional[bool]

    # Contatore resettabile delle ricerche web del giro corrente (tetto MAX_WEB_SEARCHES).
    # Viene azzerato quando una modifica HITL avvia un nuovo giro di ricerca mirata.
    web_search_count: Optional[int]

    # --- Drafting + Human-in-the-loop ---
    draft_content: Optional[str]
    human_feedback: Optional[str]
    revision_count: Optional[int]
    status: Optional[str]
