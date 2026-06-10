"""
Definizione dello stato dell'agente e schemi di pianificazione.
Soddisfa i requisiti:
- MCP / State Management: stato esplicito aggiornato ad ogni step.
- Planning: schema strutturato per la sequenza di post.
"""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from typing import List, Optional
from langgraph.graph import MessagesState


# ============================================================
# SCHEMI DI SCOPING
# User Clarification + Brief Generation
# ============================================================
class ClarifyWithUser(BaseModel):
    """
    Decisione strutturata sulla necessita' di chiarimenti. Usata nella fase di scoping per capire se servono chiarimenti o no.
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
    Brief strutturato che trasforma la richiesta dell'utente in un obiettivo editoriale chiaro per le richieste successive.
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


class KGExtraction(BaseModel):
    """
    Estrazione strutturata di conoscenza da un post approvato, per arricchire il KG
    con i 'key claims' e le 'relationships between topics' richiesti dalle specifiche.
    """
    key_claims: list[str] = Field(
        default_factory=list,
        description="2-4 affermazioni fattuali chiave estratte dall'articolo (frasi brevi e autonome)."
    )
    related_topics: list[str] = Field(
        default_factory=list,
        description="2-3 argomenti brevi correlati (1-3 parole), per collegare il post ad altri nel blog."
    )
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
    """Output del processo di pianificazione editoriale: una sequenza ordinata di post.
    con il campo reasoning che documenta il ragionamento del modello sulla scelta ed ordine dei post.
    """
    reasoning: str = Field(
        description="Ragionamento passo-passo per selezione e ordine, con attenzione a diversita' e copertura"
    )
    planned_posts: List[PostPlan] = Field(description="Sequenza ordinata dei post pianificati")


# ============================================================
# SCHEMA DI REVISIONE EDITORIALE (gate HITL dopo il planning)
# Interpreta in modo strutturato la risposta in linguaggio naturale dell'utente
# davanti alla lista di proposte. Scelta di design: invece di piu' liste separate
# (di cui una annidata), usiamo UNA SOLA lista di azioni, una per proposta menzionata,
# con un'azione esplicita (write/modify/drop). Per un modello piccolo e' molto piu'
# affidabile scegliere un enum per ogni proposta che decidere in quale lista mettere
# un indice: prima il 3B tendeva a buttare tutto in 'to_write' ignorando le modifiche.
# ============================================================
class ProposalAction(BaseModel):
    """Azione decisa dall'utente su UNA singola proposta, identificata dal suo numero."""
    index: int = Field(description="Il NUMERO della proposta come mostrato nella lista (1-based)")
    action: Literal["write", "modify", "drop"] = Field(
        description="write = scrivila cosi' com'e'; modify = cambiala (serve instruction); drop = scartala"
    )
    instruction: str = Field(
        default="",
        description="SOLO se action='modify': cosa cambiare di questa proposta (l'istruzione dell'utente)"
    )


class EditorialDecision(BaseModel):
    """Decisione editoriale dell'utente: una voce per ogni proposta che ha menzionato."""
    actions: List[ProposalAction] = Field(
        default_factory=list,
        description="Una ProposalAction per ogni proposta che l'utente vuole scrivere, modificare o scartare"
    )
    request_new: bool = Field(
        default=False,
        description="True SOLO se l'utente chiede esplicitamente nuove proposte/di rimpiazzare le scartate"
    )
    new_hint: str = Field(
        default="",
        description="Eventuale spunto specifico per le nuove proposte (es. 'confronto con la BMW M3')"
    )


# ============================================================
# INPUT / STATO (requisito "MCP / State Management")
# ============================================================
class StateInput(TypedDict):
    """Input iniziale fornito dall'utente per avviare l'agente.
    passo solo la richiesta dell'utente
    """
    
    user_input: str


class State(MessagesState):
    """
    Stato globale dell'agente cioè il "contesto" gestito esplicitamente.
    Eredita 'messages' da MessagesState, che funge anche da
    contenitore degli output dei tool.

    Mappa con i campi richiesti dalle specifiche (State Management) ed extra:
      - user_input      -> input utente
      - messages        -> tool outputs (ereditato)
      - reasoning_trace -> traccia di ragionamento (ReAct)
      - kg_summary      -> sintesi del Knowledge Graph
      - planning_info   -> informazioni di pianificazione (sequenza di post)
      - research_brief  -> brief strutturato che trasforma la richiesta dell'utente in un obiettivo editoriale chiaro
      - current_topic   -> argomento del post corrente
      - trends_summary  -> sintesi dei trend RSS
      - sources         -> fonti trovate per il post corrente
      - local_sources   -> fonti locali trovate per il post corrente
      - raw_notes       -> note grezze raccolte durante la ricerca
      - forced_web_search-> flag del guardrail "verifica fonti": True dopo che il sistema ha forzato una
      ricerca web perche' il modello stava per scrivere senza alcuna fonte. Evita i loop.
      - web_search_count -> contatore delle ricerche web del giro corrente (tetto MAX_WEB_SEARCHES).
      - draft_content    -> bozza del post corrente
      - human_feedback   -> feedback umano sulla bozza
      - revision_count   -> contatore delle revisioni
      - status           -> stato del post corrente
      
    Quasi tutti i campi sono optional, tranne lo user_input e vengono popolati all'avanzare dei nodi nel grafo.
    """
    # Input
    user_input: str

    # Scoping
    research_brief: Optional[str]
    # Conteggio dei giri di chiarimento (per evitare loop infiniti di domande).
    clarification_count: Optional[int]

    # Pianificazione / topic
    current_topic: Optional[str]
    planning_info: Optional[List[dict]]

    # --- Pianificazione multi-post (gate editoriale + ciclo di scrittura) ---
    # Numero massimo di post richiesto dall'utente (dinamico, estratto dalla richiesta).
    # Serve al refill: se scarto proposte e scendo sotto questo numero, l'agente puo'
    # propormene di nuove per tornare a questo target.
    num_posts_requested: Optional[int]
    # Coda dei post che l'utente ha scelto di scrivere (lista di PostPlan come dict).
    # Viene consumata un post alla volta dal ciclo di scrittura.
    selected_posts: Optional[List[dict]]
    # Il PostPlan del post attualmente in scrittura (per leggere categoria/giustificazione
    # corrette, invece di assumere sempre planning_info[0]).
    current_post: Optional[dict]
    # Topic gia' scartati dall'utente nel gate editoriale: il refill li evita per non
    # riproporre cose che l'utente ha gia' rifiutato.
    rejected_topics: Optional[List[str]]

    # Knowledge Graph
    kg_summary: Optional[str]

    # Trend RSS
    trends_summary: Optional[str]

    # Ragionamento ReAct
    reasoning_trace: Optional[str]

    # Grounding / citazioni (K-RAG)
    sources: Optional[List[str]]

    # Documenti locali recuperati deterministicamente nel K-RAG (research_agent_node),
    # passati al drafting come fonti citabili.
    local_sources: Optional[List[str]]

    # Note grezze (osservazioni complete dei tool) raccolte durante la ricerca.
    # Conserviamo l'output integrale
    # dei tool oltre al riassunto, utile in fase di drafting/modifica per recuperare
    # dettagli che il riassunto potrebbe aver tagliato.
    raw_notes: Optional[List[str]]

    # Flag del guardrail "verifica fonti": True dopo che il sistema ha forzato una
    # ricerca web perche' il modello stava per scrivere senza alcuna fonte. Evita i loop.
    forced_web_search: Optional[bool]

    # Contatore resettabile delle ricerche web del giro corrente (tetto MAX_WEB_SEARCHES).
    # Viene azzerato quando una modifica HITL avvia un nuovo giro di ricerca mirata.
    web_search_count: Optional[int]

    # Drafting + Human-in-the-loop
    draft_content: Optional[str]
    human_feedback: Optional[str]
    revision_count: Optional[int]
    status: Optional[str]
