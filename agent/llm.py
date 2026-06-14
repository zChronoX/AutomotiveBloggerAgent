"""
Inizializzazione dei client LLM locali con Ollama.
L'archiettura usa lo stesso modello per le varie fasi:
  - llm (Ministral): il "cervello" — planning, scelta dei tool, ricerca, grading,
    routing, update KG. Veloce e affidabile nella scelta dei tool.
  - drafting_llm (Ministral): la "penna" — Solo per la stesura finale dell'articolo.
    Produce prosa piu' ricca e discorsiva, adatta a un blog.
"""

from langchain_ollama import ChatOllama
from config.settings import Configuration

config = Configuration()

llm = ChatOllama(
    model=config.model_name,
    temperature=config.temperature,
    num_ctx=config.model_num_ctx,
    keep_alive=0,
    num_gpu = 98,
)

drafting_llm = ChatOllama(
    model=config.draft_model_name,
    temperature=config.draft_temperature,
    num_ctx=config.draft_num_ctx,
    # Qui uso il massimo numero di token in scrittura per evitare che il modello inizi a scrivere loop di cose
    num_predict=config.draft_num_predict,  
    keep_alive=0,
    num_gpu = 98,
)
