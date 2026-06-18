"""
Definizione e compilazione del grafo LangGraph.
Assemblaggio nodi, edges e compilare.
"""


#Import degli stati del grafo, START è per iniziare, END è il punto d'uscita.
from langgraph.graph import StateGraph, END, START
#Il memory saver è importante perché permette all'agente di ricordare
#lo stato della conversazione da un punto all'altro.
from langgraph.checkpoint.memory import MemorySaver

from agent.state import State, StateInput
from agent.nodes import (
    clarification_node,
    brief_node,
    kg_context_node,
    planner_node,
    editorial_review_node,
    suggest_topics_node,
    research_agent_node,
    revision_research_node,
    forced_search_node,
    resilient_tool_node,
    rewrite_question_node,
    drafting_node,
    review_node,
    update_kg_node,
    next_post_node,
)
from agent.routing import (
    route_after_clarification, route_after_planner,
    route_after_research, grade_documents,
)

#Metodo che inizializza il grafo
def build_graph(checkpointer="default"):
    """Costruisce e compila il grafo dell'agente blogger.
    checkpointer:
      - "default" -> usa un MemorySaver interno (uso da CLI con main.py).
      - None      -> compila SENZA checkpointer: necessario per LangGraph Studio,
                     che inietta il PROPRIO checkpointer persistente. Passargli un
                     grafo gia' checkpointato darebbe conflitto.
      - oggetto   -> usa il checkpointer fornito.
    """
    workflow = StateGraph(State, input=StateInput)

    # Nodi
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("brief_node", brief_node)
    workflow.add_node("kg_context", kg_context_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("editorial_review_node", editorial_review_node)
    workflow.add_node("suggest_topics_node", suggest_topics_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("revision_research_node", revision_research_node)
    workflow.add_node("forced_search_node", forced_search_node)
    workflow.add_node("tools", resilient_tool_node)
    workflow.add_node("rewrite_question_node", rewrite_question_node)
    workflow.add_node("drafting_node", drafting_node)
    workflow.add_node("review_node", review_node)
    workflow.add_node("update_kg_node", update_kg_node)
    workflow.add_node("next_post_node", next_post_node)
    workflow.add_edge(START, "clarification_node")
    workflow.add_conditional_edges("clarification_node", route_after_clarification)
    workflow.add_edge("brief_node", "kg_context")
    workflow.add_edge("kg_context", "planner")
    workflow.add_conditional_edges("planner", route_after_planner)
    workflow.add_conditional_edges("research_agent", route_after_research)
    workflow.add_conditional_edges("revision_research_node", route_after_research)
    workflow.add_edge("forced_search_node", "tools")
    workflow.add_conditional_edges("tools", grade_documents)
    workflow.add_edge("rewrite_question_node", "research_agent")

    workflow.add_edge("drafting_node", "review_node")
    if checkpointer == "default":
        return workflow.compile(checkpointer=MemorySaver())
    if checkpointer is None:
        # Compilo senza checkpoiter per LangGraph Studio sennò mi da problemi
        return workflow.compile()
    return workflow.compile(checkpointer=checkpointer)


# Istanza per costruire il grafico con la memoria quando lo uso da terminiale
graph = build_graph()



def make_studio_graph():
    """Entry point per LangGraph Studio (grafo senza checkpointer interno)."""
    return build_graph(checkpointer=None)
