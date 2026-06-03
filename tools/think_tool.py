"""
Tool di riflessione strategica (think_tool).

Ripreso dal notebook 2 del tutorial Deep Research di LangGraph (deep_research_from_scratch).
Adattato al dominio del blog automotive e alla lingua italiana.

Scopo: rendere ESPLICITO il passo "Thought" del ciclo ReAct come una vera tool-call
tracciabile in LangSmith. Invece di lasciare il ragionamento solo nel testo del modello
(che Granite/altri modelli a volte non verbalizzano), il modello chiama think_tool per
fermarsi, analizzare cosa ha trovato e pianificare il passo successivo.

Questo soddisfa direttamente il requisito delle specifiche:
"explicit reasoning steps (e.g., Thought -> Action -> Observation)".
"""

from langchain_core.tools import tool
from prompts.tool_prompts import THINK_TOOL_PROMPT


@tool(description=THINK_TOOL_PROMPT)
def think_tool(reflection: str) -> str:
    """Strumento di riflessione strategica sui progressi della ricerca.

    Usalo DOPO ogni ricerca per analizzare i risultati e pianificare i passi successivi.
    Crea una pausa deliberata nel flusso di ricerca per decidere con consapevolezza.

    Quando usarlo:
    - Dopo aver ricevuto risultati di ricerca: quali informazioni chiave ho trovato?
    - Prima di decidere il prossimo passo: ho abbastanza per scrivere un buon post?
    - Quando valuti i gap: quali informazioni specifiche mi mancano ancora?
    - Prima di concludere la ricerca: posso fornire una risposta completa ora?

    La riflessione dovrebbe affrontare:
    1. Analisi dei risultati attuali - quali informazioni concrete ho raccolto?
    2. Valutazione dei gap - quali informazioni cruciali mancano ancora?
    3. Valutazione della qualita' - ho prove/esempi sufficienti per un buon articolo?
    4. Decisione - continuare la ricerca o procedere alla stesura?

    Args:
        reflection: la tua riflessione dettagliata su progressi, gap e prossimi passi.

    Returns:
        La riflessione registrata, per tenerne traccia nel flusso di ragionamento.
    """
    return f"Riflessione registrata: {reflection}"
