""" 
Modulo che gestisce tutti i tool del sistema, in cui
decidiamo quale tool devono essere scelti dal modello
e quali devono essere chiamati per forza
"""

from typing import List, Dict
from langchain_core.tools import BaseTool

# Questi sono i tool che vengono scelti in base alla richiesta e vengono
# passati al research_agent. Escludiamo i tool dei documenti locali,
# contesto editoriale, update del KG, immagini e SEO.
def get_react_tools() -> List[BaseTool]:
    from tools.mcp_client_tool import mcp_web_search          # Ricerca
    from tools.kg_tool import query_knowledge_graph           # Knowledge Graph
    from tools.specs_tool import fetch_vehicle_specs          # Specifiche veicoli
    from tools.compare_vehicles_tool import compare_vehicles_tool  # Comparazione veicoli
    from tools.trend_tool import fetch_automotive_trends      # Trend attuali
    from tools.think_tool import think_tool                   # Think (ReAct)

    return [
        mcp_web_search,
        query_knowledge_graph,
        fetch_vehicle_specs,
        compare_vehicles_tool,
        fetch_automotive_trends,
        think_tool,
    ]


# Tutti i tool che vengono passati al resilient_tool_node. 
def get_all_tools() -> List[BaseTool]:
    from tools.rag_tool import retrieve_local_documents
    from tools.kg_tool import (
        update_knowledge_graph,
        list_blog_topics,
        get_editorial_context,
    )
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

# Torno i tool in base al nome
def get_tools_by_name() -> Dict[str, BaseTool]:
    return {tool.name: tool for tool in get_all_tools()}