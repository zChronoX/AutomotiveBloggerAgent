"""
Conditional edges e logica di routing del grafo.
Estratte da blogger_agent.py per separare la logica decisionale dai nodi.
"""

from typing import Literal
from langchain_core.messages import HumanMessage, ToolMessage

from .helpers import wants_post, has_collected_sources
from .llm import llm
from utils import GradeDocuments, should_grade_tool

# Limite massimo di passi di ricerca per post. Con 11 tool disponibili e task complessi
# (es. confronti che richiedono fetch_vehicle_specs su 2 veicoli + compare_vehicles),
# 6 e' un buon equilibrio: lascia margine per la sequenza ottimale senza permettere
# loop infiniti. Oltre questo limite il sistema passa comunque alla stesura.
MAX_RESEARCH_STEPS = 10


def route_after_clarification(state: dict) -> Literal["clarification_node", "brief_node"]:
    """
    Dopo il nodo di clarification:
      - status 'clarifying' -> l'utente ha risposto a una domanda: torna a clarification_node
        per RI-VALUTARE se ora la richiesta e' chiara (il loop e' limitato da MAX_CLARIFICATIONS).
      - status 'scoped' (o altro) -> la richiesta e' chiara: procedi alla generazione del brief.
    """
    if state.get("status") == "clarifying":
        return "clarification_node"
    return "brief_node"


def route_after_planner(state: dict) -> Literal["research_agent", "suggest_topics_node"]:
    """L'utente vuole un POST scritto, o solo dei SUGGERIMENTI di argomenti?"""
    return "research_agent" if wants_post(state.get("user_input", "")) else "suggest_topics_node"


def route_after_research(state: dict) -> Literal["tools", "drafting_node", "forced_search_node"]:
    """Tool chiamato -> esecuzione; altrimenti -> stesura (con guardia anti-loop e guardrail fonti)."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        n = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if n >= MAX_RESEARCH_STEPS:
            print("[ReAct] Max passi di ricerca raggiunto: passo alla stesura.")
            return "drafting_node"
        return "tools"
    # Il modello vuole passare alla stesura. GUARDRAIL: se non ha raccolto NESSUNA fonte
    # e non l'abbiamo gia' forzata, imponiamo una ricerca web prima di scrivere.
    if not has_collected_sources(state) and not state.get("forced_web_search"):
        return "forced_search_node"
    return "drafting_node"


def grade_documents(state: dict) -> Literal["research_agent", "rewrite_question_node"]:
    """Self-RAG: valuta SOLO le fonti dei tool di ricerca (via should_grade_tool)."""
    last = state["messages"][-1]
    if not isinstance(last, ToolMessage) or not should_grade_tool(getattr(last, "name", "")):
        return "research_agent"

    print(f"\n[Self-RAG] Valuto la rilevanza delle fonti dal tool '{last.name}'...")
    grader = llm.with_structured_output(GradeDocuments)
    prompt = (
        f"Il seguente documento e' rilevante per il tema '{state.get('current_topic','')}'?\n"
        f"Documento: {last.content}\nRispondi solo 'yes' o 'no'."
    )
    try:
        score = grader.invoke([HumanMessage(content=prompt)]).binary_score
    except Exception:
        return "research_agent"

    if score == "yes":
        print("-> Fonti rilevanti.")
        return "research_agent"
    print("-> Fonti non rilevanti: riformulo.")
    return "rewrite_question_node"
