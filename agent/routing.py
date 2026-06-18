"""
Logica di routing del grafo, cioè decido in quale nodo far andare
l'agente a seconda dello stato in cui si trova.
"""

from typing import Literal
from langchain_core.messages import HumanMessage, ToolMessage

from .helpers import wants_post, has_collected_sources, canonical_topic
from .llm import llm
from utils import GradeDocuments, should_grade_tool

# Limite massimo di chiamate per i tool durante l'intero ciclo, serve per evitare che
# il modello chiami un numero di tool infinito e impieghi troppo tempo.
# 10 è un numero accettabile, in quanto considera il caso "peggiore"
# ovvero quando l'agente chiama tutti i tool a disposizione, più eventuali 2 ricerche web.
MAX_RESEARCH_STEPS = 10



#Semplicemente indirizzo l'agente al brief se la richiesta è chiara, sennò resto nel nodo clarification.
def route_after_clarification(state: dict) -> Literal["clarification_node", "brief_node"]:
    """
    Controllo il campo "status", se la riposta è chiara (oppure siamo arrivati al limite dei 2 chiarimenti)
    allora andiamo al nodo per il brief, altrimenti chiediamo più spiegazioni.
    Il numero di chiarimenti è impostato a 2 nel nodo di clarification. Empiricamente
    dovrebbe essere un buon compromesso tra numero di chiarimenti ed informazioni raccolte per il briefing.
    """
    if state.get("status") == "clarifying":
        return "clarification_node"
    return "brief_node"



# Qui serve per modificare la rotta del grafo nel caso in cui l'utente vuole solo
# suggerimenti e quindi glieli diamo ma non scriviamo post. Altrimenti se vuole
# un post andiamo avanti: non più direttamente alla ricerca, ma al nodo editoriale
# (editorial_review_node), dove l'utente sceglie quali post pianificati scrivere,
# ne modifica alcuni o ne chiede di nuovi.
def route_after_planner(state: dict) -> Literal["editorial_review_node", "suggest_topics_node"]:
    """L'utente vuole dei post scritti, va al nodo editoriale, altrimenti va a suggest_topics_node."""
    return "editorial_review_node" if wants_post(state.get("user_input", "")) else "suggest_topics_node"




# Qui gestisco il ciclo ReACT in modo più complesso con una serie di guardrail per le fonti.
# Come già evidenziato più volte, il modello locale non sempre segue i passaggi del ReACT
# (dice di voler chiamare 3 tool, ma alla fine non ne chiama mezzo, oppure chiama un tool totalmente sbagliato)
# pertanto qui vediamo se il numero di tool totali usati è 10 (max_step di sopra), se si allora forziamo il passaggio alla stesura, altrimenti procediamo normalmente.
# Se non ha usato ancora 10 tool, ma ha comunque fonti (perché c'è stata una ricerca oppure perché ha chiamato uno dei tool grounding)
# allora andiamo comunque con la stesura. Altrimenti forziamo la ricerca web. 

def route_after_research(state: dict) -> Literal["tools", "drafting_node", "forced_search_node"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        n = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if n >= MAX_RESEARCH_STEPS:
            print("Max passi di ricerca raggiunto, si passa alla stesura.")
            return "drafting_node"
        return "tools"
    # Il modello vuole passare alla stesura. Usiamo un guardrail: se non ha raccolto nessuna fonte
    # dai tool grounding, e non l'abbiamo gia' forzata, imponiamo una ricerca web prima di scrivere.
    if not has_collected_sources(state) and not state.get("forced_web_search"):
        return "forced_search_node"
    return "drafting_node"




# Valuto la pertinenza dei risultati raccolti dai tool "gradabili", come la ricerca web
# che spesso porta a fonti poco pertinenti anche con query corrette. E' una whitelist
# perché nel caso in cui aggiungessi altri tool futuri che non hanno bisogno di essere gradati
# di default non viene gradato.

def grade_documents(state: dict) -> Literal["research_agent", "rewrite_question_node"]:
    last = state["messages"][-1]
    if not isinstance(last, ToolMessage) or not should_grade_tool(getattr(last, "name", "")):
        return "research_agent"

    content = str(getattr(last, "content", ""))

    # I messaggi di servizio (limite ricerche raggiunto, chiamata ripetuta bloccata) non
    # sono fonti da valutare: valutarli produceva un terzo "Fonti non rilevanti: riformulo"
    # che spingeva il modello a ritentare ricerche ormai bloccate. Si prosegue e basta.
    if content.startswith("Limite di") or content.startswith("Il tool '"):
        return "research_agent"

    print(f"\nValuto la rilevanza delle fonti dal tool '{last.name}'.")

    # Soggetto pulito per il confronto: il current_topic può essere un titolo lungo di
    # proposta; la chiave canonica (es. "audi rs3") rende il giudizio più stabile ed evita
    # di scartare fonti valide solo perché non combaciano con tutta la frase.
    subject = canonical_topic(state.get("current_topic", "")) or (state.get("current_topic", "") or "")[:80]

    #Il grader usato è sempre ministral 3, ma in questo caso gli sto chiedendo di rispondere in modo
    #strutturato attraverso un prompt ingegnerizzato, in cui mi dovrà rispondere semplicemente si o no
    grader = llm.with_structured_output(GradeDocuments)

    # Il giudizio di valutazione delle fonti è sulla qualità delle fonti trovate
    # inerenti a quel veicolo oppure all'argomento trattato, se tra le fonti
    # c'è spazzatura, come immagini, video, link e roba inutile dal punto di vista
    # testuale, viene considerato come non valido e quindi viene scartato.
    prompt = (
        f"Devi decidere se questo risultato di ricerca è utile per scrivere un articolo "
        f"tecnico sul veicolo/tema '{subject}'.\n"
        f"Rispondi 'yes' SOLO se contiene informazioni concrete e utili: specifiche, prova su "
        f"strada, dati di prestazioni/consumi/prezzi/sicurezza, analisi o recensione vera.\n"
        f"Rispondi 'no' se è perlopiù navigazione del sito, elenchi di gallerie o foto, "
        f"titoli ripetuti, chiacchiere o post social, pagine di soli video/link, senza reale "
        f"contenuto informativo sul veicolo.\n"
        f"Documento:\n{content}\n\nRispondi solo 'yes' o 'no'."
    )
    try:
        score = grader.invoke([HumanMessage(content=prompt)]).binary_score
    except Exception:
        return "research_agent"

    if score == "yes":
        print(": Fonti rilevanti.")
        return "research_agent"
    print(": Fonti non rilevanti: riformulo.")
    return "rewrite_question_node"
