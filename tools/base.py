from typing import List, Dict
from langchain_core.tools import BaseTool


def get_react_tools() -> List[BaseTool]:
    """
    Tool che il modello sceglie DINAMICAMENTE durante il ciclo ReAct.

    Sono solo i tool SITUAZIONALI, la cui scelta dipende dal contesto della richiesta
    (cercare sul web? consultare le specifiche? fare un confronto? cercare trend?).
    Esporre meno tool al modello (5 invece di 11) riduce drasticamente gli errori di
    tool selection con i modelli locali piccoli (3B), senza sacrificare la dynamic
    selection richiesta dalle specifiche.

    I tool ESCLUSI da qui sono invocati in modo DETERMINISTICO nei nodi del grafo,
    perche' le specifiche li rendono obbligatori in fasi precise (non sono decisioni
    contestuali):
      - list_blog_topics / get_editorial_context -> KG (topic suggestion / drafting)
      - retrieve_local_documents                 -> RAG (K-RAG in research_agent_node)
      - update_knowledge_graph                    -> KG update (dopo approvazione HITL)
      - generate_cover_image / analyze_seo_and_readability -> pubblicazione
    """
    from tools.mcp_client_tool import mcp_web_search          # Core: Search
    from tools.kg_tool import query_knowledge_graph           # Core: KG (lettura cronologia)
    from tools.specs_tool import fetch_vehicle_specs          # Team
    from tools.compare_vehicles_tool import compare_vehicles_tool  # Team
    from tools.trend_tool import fetch_automotive_trends      # Team
    from tools.think_tool import think_tool                   # Ripreso dal notebook 2 Deep Research

    return [
        mcp_web_search,
        query_knowledge_graph,
        fetch_vehicle_specs,
        compare_vehicles_tool,
        fetch_automotive_trends,
        think_tool,
    ]


def get_all_tools() -> List[BaseTool]:
    """
    TUTTI i tool del sistema (ReAct + deterministici). Usata dal resilient_tool_node
    per il lookup quando deve eseguire un tool (qualunque sia stato chiamato).

    Mappa con i requisiti di tooling delle specifiche:
      - 3 TOOL CORE obbligatori:
          * Search          -> mcp_web_search
          * RAG retrieval    -> retrieve_local_documents
          * Knowledge Graph  -> query_knowledge_graph / update_knowledge_graph
                                (+ estensioni di lettura: list_blog_topics, get_editorial_context)
      - TOOL PROGETTATI DAL TEAM (>= 2 richiesti, qui 5):
          * generate_cover_image, analyze_seo_and_readability,
            fetch_automotive_trends, fetch_vehicle_specs, compare_vehicles_tool
    """
    # --- Core: RAG ---
    from tools.rag_tool import retrieve_local_documents
    # --- Core: Knowledge Graph (update + letture deterministiche) ---
    from tools.kg_tool import (
        update_knowledge_graph,
        list_blog_topics,
        get_editorial_context,
    )
    # --- Tool del team (deterministici alla pubblicazione) ---
    from tools.image_tool import generate_cover_image
    from tools.seo_tool import analyze_seo_and_readability

    return get_react_tools() + [
        retrieve_local_documents,
        update_knowledge_graph,
        list_blog_topics,
        get_editorial_context,
        generate_cover_image,
        analyze_seo_and_readability,
    ]


def get_tools_by_name() -> Dict[str, BaseTool]:
    """Dizionario {nome_tool: tool} comodo per lookup diretto."""
    return {tool.name: tool for tool in get_all_tools()}