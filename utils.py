"""
Utility condivisa per il meccanismo del Self-RAG dell'agente.
Decide quale output dei tool vanno valutati per qualità/rilevanza prima 
di essere usati nei post.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional

# Quando l'agente deve valutare se una fonte è rilevante per l'argomento o no.
# Forzo il modello a scegliere esattamente tra si o no.
class GradeDocuments(BaseModel):
    """Valuta la rilevanza dei documenti recuperati con un punteggio binario."""
    binary_score: Literal["yes", "no"] = Field(
        description="Punteggio di rilevanza: 'yes' se il documento e' rilevante, 'no' se non lo e'"
    )




# Whitelist di tools da sottoporre al Self-RAG.
# In pratica inseriamo qui i tool di ricerca di fonti esterne.
# Escludiamo tool come retrieve_local_documents che recuperano dati da fonti curate.
# Il fetch_vehicle_specs, a differenza del web search, restituisce sempre dati fattuali e verificati
# quindi non ha senso sottoporlo al Self-RAG. Tra le fonti escludiamo anche i trend RSS che non sono fonti ma ispirazione
# per il brainstorming.
# I tool del KG perché riguardano letture e scritture interne al grafo e non c'è nulla da gradare
# e i tool di generazione immagini e SEO che vengono usati solo alla fine per arricchire il post.

GRADABLE_TOOLS = {
    "mcp_web_search",
}


# Funzione che viene passata ai nodi del grafo per decidere se un tool va sottoposto al Self-RAG.
def should_grade_tool(tool_name: Optional[str]) -> bool:
    if not tool_name:
        return False
    return tool_name.strip().lower() in GRADABLE_TOOLS