"""
Inizializzazione dei client LLM (Ollama).

Architettura ibrida a due modelli:
  - llm (Granite): il "cervello" — planning, scelta dei tool, ricerca, grading,
    routing, update KG. Veloce e affidabile nella scelta dei tool.
  - drafting_llm (Ministral): la "penna" — SOLO la stesura dell'articolo finale.
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
)

drafting_llm = ChatOllama(
    model=config.draft_model_name,
    temperature=config.draft_temperature,
    num_ctx=config.draft_num_ctx,
    keep_alive=0,
)
