from pydantic import BaseModel, Field
from typing import Literal, Optional

# --- MODELLI STRUTTURATI PER L'LLM ---

class GradeDocuments(BaseModel):
    """Valuta la rilevanza dei documenti recuperati con un punteggio binario."""
    binary_score: Literal["yes", "no"] = Field(
        description="Punteggio di rilevanza: 'yes' se il documento e' rilevante, 'no' se non lo e'"
    )


# --- SELF-RAG: QUALI TOOL PRODUCONO FONTI DA VALUTARE ---

# Whitelist ESPLICITA dei tool il cui output e' una FONTE che verra' citata nel
# post e che quindi deve passare dal controllo di rilevanza/qualita' (Self-RAG).
#
# Razionale (utile anche per la relazione):
#   - mcp_web_search          -> ricerca web esterna: unica fonte dal web aperto,
#                                va gradata perche' i risultati possono essere
#                                off-topic o di bassa qualita'.
#
# Volutamente ESCLUSI (non sono "fonti" da gradare col Self-RAG):
#   - retrieve_local_documents-> RAG locale: i documenti sono CURATI dal blogger
#                                (appunti, manuali, vecchi articoli), quindi sono
#                                inherentemente rilevanti per il dominio del blog.
#                                Il filtro per distanza in retriever.py (DISTANCE_THRESHOLD)
#                                scarta gia' i chunk non pertinenti. Gradarli di nuovo
#                                col Self-RAG (Granite 3B) introduce falsi negativi
#                                (il modello piccolo non matcha il topic elaborato del
#                                planner col contenuto tecnico dei documenti locali).
#   - fetch_vehicle_specs     -> scheda tecnica (API Ninjas + Wikipedia): fonte FATTUALE
#                                per il veicolo richiesto, inherentemente rilevante.
#   - fetch_automotive_trends -> ispirazione/brainstorming, non grounding dell'articolo.
#   - compare_vehicles        -> output strutturato del modello fine-tuned, non da gradare.
#   - query_knowledge_graph / list_blog_topics / get_editorial_context / update_knowledge_graph
#                             -> letture/scritture del KG, non documenti esterni da valutare.
#   - generate_cover_image / analyze_seo_and_readability
#                             -> output ausiliari, non fonti informative.
#
# Usare una whitelist (e non una blacklist) rende il default SICURO: qualsiasi
# nuovo tool non elencato qui NON viene gradato finche' non lo aggiungiamo apposta.
GRADABLE_TOOLS = {
    "mcp_web_search",
}


# --- FUNZIONI DI SUPPORTO ---

def should_grade_tool(tool_name: Optional[str]) -> bool:
    """
    Restituisce True solo se l'output del tool e' una fonte da valutare col Self-RAG.

    Robusta a input anomali: se tool_name e' None o vuoto restituisce False
    (default sicuro: non gradare). Il confronto e' case-insensitive.
    """
    if not tool_name:
        return False
    return tool_name.strip().lower() in GRADABLE_TOOLS