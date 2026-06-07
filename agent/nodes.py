"""

Cervello dell'applicazione, contiene le definizioni di tutti i nodi del grafo di LangGraph.
Ogni nodo è un funzione di python che prende lo stato corrente come input, fa l'operazione 
specifica e restituisce il nuovo stato, per poi essere passato ai nodi successivi.

Abbiamo un totalità di 6 fasi (FASE 0 - FASE 5)

FASE 0 - Scoping: questa fase si ocupa di chiarire le richieste ambigue dell'utente. In poche parole 
l'agente per evitare di scegliere un topic casuale, chiede maggiori informazioni all'utente, solo
nel caso in cui la richiesta non è vaga. Altrimenti si fa direttamente alla Fase 1. Si usa
lo HITL (Human-in-the-loop) per permettere all'utente di rispondere.

FASE 1 - KG: questa fase si occupa di interrogare il KG per capire quali sono i topic che potrebbero
interessare il pubblico del blog. Viene usata la tool kg_topics_overview() per ottenere i topic più popolari, 
nonchè viene effetuata una gap analysis per capire quali sono i topic che non sono stati coperti dal blog.
Successivamente si formula una breve sintesi dei topic che potrebbero interessare l'utente.
Usa anche il tool RSS per capire le notizie fresce di giornata.

FASE 2 - Planning e suggerimenti: questa fase si occupa di pianificare la stesura dell'articolo. 
Viene creato il piano editoriale, in cui al modello viene passato il Brief (frutto dello scoping + chiarimenti)
la lista dei gap del KG e i trend RSS.

FASE 3 - Research (ReAct + Self-RAG): Fase più importante del ciclo di vita dell'agente. 
Rappresenta il punto in cui l'agente ragiona, in base alla richiesta ottenuta ed esegue scelte precise.
Inizialmente viene chiesto al KG il contsto del topic, dei topic correlati e usa i correlati
per espandere la query verso ChromaDB per ottenere i chunk dei documenti utili.
Dopo di che, passa tutto al ReAct, ini cui il modello decide in autonomia che tool usare
in base al contesto e come definire i parametri dei vari tool per ottere le informazioni necessarie per i post.

Questa fase viene richiamata anche quando c'è una revisione, nel caso in cui l'utente decide di aggiungere qualcosa
ad esempio in una recensione anche il confronto con un modello specifico, si torna a questa fase per raccogliere più informazioni.
Inoltre vengono valutate anche la pertinenza delle fonti raccolte, nel caso in cui non lo sono, viene rifatta una ricerca.


FASE 4 - Drafting: Fase quasi finale in cui si prendono tutte le "raw_notes" quindi le informazioni
raccolte dai vari tool (web, RAG, ecc), il contesto del KG e vengono passati al modello per la stesura
dell'articolo finale.
Alla fine di questa fase viene presentata la bozza all'utente che decide se approvarla o riscriverla.
La riscrittura, differentemente dalla fase sopra, agisce solo sul testo e non sulle fonti. Ad esempio
se chiedo di scrivere con un lessico più semplice, non ho bisogno di tornare nella fase di ricerca sopra.

FASE 5 - Aggiornamento del KG: Una volta che la bozza rispetta le specifiche dell'utente e questo
la approva, viene salvata nel KG, collegata al topic e vengono generate le immagini e l'analisi SEO.

"""

import re
import time
from typing import Literal

from langgraph.types import interrupt, Command
from langgraph.graph import END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .llm import llm, drafting_llm
from .state import PlanningSchema, ClarifyWithUser, ResearchBrief, KGExtraction
from .helpers import (
    trace, has_collected_sources, strip_reasoning_preamble, normalize_tool_args,
    wants_post, modification_needs_research, is_clearly_vague, canonical_topic,
)
from config.settings import Configuration
from prompts.system_prompts import (
    blogger_system_prompt, default_background, default_editorial_guidelines,
)
from prompts.agent_prompts import (
    PLANNING_PROMPT, RESEARCH_KICKOFF, DRAFT_PROMPT,
    CLARIFICATION_PROMPT, BRIEF_PROMPT, KG_EXTRACTION_PROMPT,
)
from utils import should_grade_tool
from tools.base import get_all_tools, get_react_tools
# Funzioni pure del KG chiamate in modo deterministico nelle fasi obbligatorie
from knowledge_graph.queries import kg_topics_overview, kg_topic_context, kg_related_topics
from knowledge_graph.updater import update_kg_data
# Retrieval deterministico per il K-RAG (sia KG che RAG)
from rag.retriever import retrieve_local
# Tool di arricchimento eseguiti deterministicamente alla pubblicazione (post-approvazione)
from tools.image_tool import generate_cover_image
from tools.seo_tool import analyze_seo_and_readability

config = Configuration()
tools = get_all_tools()                  # tutti i tool per lookup nel resilient_tool_node
react_tools = get_react_tools()          # solo i 5 tool situazionali per il bind_tools
llm_with_tools = llm.bind_tools(react_tools)
tools_by_name = {t.name: t for t in tools}

KG_UPDATE_TOOL_NAME = "update_knowledge_graph"

# Limite di giri di chiarimento, per evitare che l'agente continui a chiedere all'infinito
MAX_CLARIFICATIONS = 2


# Fase di Scoping (FASE 0: Scoping + Brief)
def clarification_node(state: dict):
    """
    Valuta con structured output se la richiesta dell'utente e' chiara o vaga.
    Se vaga (e non abbiamo gia' chiesto troppe volte), si ferma con interrupt() e
    chiede un chiarimento all'utente (Human-in-the-loop). Se chiara, prosegue al brief.
    """
    msgs = state.get("messages", [])
    user_input = state.get("user_input") or (msgs[-1].content if msgs else "")
    clarif_count = state.get("clarification_count") or 0

    # Se abbiamo gia' raggiunto il limite di chiarimenti, procediamo comunque (evito i loop infinit)
    if clarif_count >= MAX_CLARIFICATIONS:
        print("[FASE 0 - SCOPING] Limite chiarimenti raggiunto: procedo con quanto ho.")
        return {"user_input": user_input, "status": "scoped"}


    # Alcune richieste sono palesemente generiche quindi forziamo sempre il chiarimento
    # a prescindere da cosa il modello vuole fare. Questo "guardrail" è stato inserito
    # per evitare che il modello proceda con una richiesta vaga e non chiarisca per niente.
    # Essendo un modello piccolo, andiamo sul sicuro in determinati casi, per altri
    # facciamo decidere a lui nella speranza che faccia la cosa giusta.
    # Questo guardrail è applicato solo al primo passaggio, difatti se l'utente ha già
    # fornito un chiarimento, non viene rivalutato (altrimenti la richiesta originale
    # generica la farebbe riscattare, stampando messaggi fuorvianti): lasciamo decidere il
    # modello sulla richiesta ormai arricchita, ma non dovrebbe andare così
    # perché solitamente il modello tende a non chiedere quasi mai.
    forced_vague = is_clearly_vague(user_input) if clarif_count == 0 else False

    if forced_vague:
        decision_need = True
        decision_question = (
            "La tua richiesta e' un po' generica. Su cosa ti piacerebbe scrivere? "
            "Per esempio: la recensione di un modello specifico (auto o moto), una guida "
            "tecnica (es. manutenzione, freni, motorizzazioni, batterie), oppure un confronto tra due veicoli?"
        )
        decision_verification = ""
        decision_reason = "deterministica"
    else:
        clarifier = llm.with_structured_output(ClarifyWithUser)
        try:
            decision = clarifier.invoke([SystemMessage(content=CLARIFICATION_PROMPT.format(user_input=user_input))])
            decision_need = decision.need_clarification
            decision_question = decision.question
            decision_verification = decision.verification
            decision_reason = "modello"
        except Exception as e:
            # Se lo structured output fallisce, non blocchiamo il flusso
            print(f"[FASE 0 - SCOPING] Clarification non riuscita ({e}): procedo senza chiarimenti.")
            return {"user_input": user_input, "status": "scoped"}

    if decision_need:
        # Interrupt HITL che usa lo stesso meccanismo della revisione 
        request = {
            "action_request": {
                "action": "clarify_request",
                "args": {},
            },
            "config": {
                "allow_accept": False, "allow_respond": True,
                "allow_ignore": True, "allow_edit": False,
            },
            "description": decision_question,
        }
        answer = interrupt(request)
        rtype = answer.get("type") if isinstance(answer, dict) else None

        # A questo punto l'interrupt si e' risolto: logghiamo la decisione una volta sola
        # (il codice sopra l'interrupt si riesegue alla ripresa, qui sotto no).
        reason_lbl = "regola deterministica" if decision_reason == "deterministica" else "valutazione del modello"
        print(f"\nRichiesta giudicata vaga ({reason_lbl}): chiesto chiarimento.")

        if rtype == "ignore":
            # L'utente non vuole chiarire: procediamo con la richiesta originale
            print("\nNessun chiarimento fornito: procedo con la richiesta originale.")
            return {"user_input": user_input, "status": "scoped"}

        # L'utente ha risposto: arricchiamo lo user_input col chiarimento e ri-valutiamo
        extra = (answer.get("args", "") if isinstance(answer.get("args"), str)
                 else str(answer.get("args", "")))
        enriched = f"{user_input}\nChiarimento dell'utente: {extra}"
        print("\nChiarimento ricevuto, rivaluto la richiesta.")
        return {
            "user_input": enriched,
            "clarification_count": clarif_count + 1,
            "status": "clarifying",
            "reasoning_trace": trace(state, f"\nChiarimento richiesto e ricevuto: {extra[:120]}"),
        }

    # Richiesta chiara: registriamo la verifica e proseguiamo
    print(f"\nRichiesta chiara: {decision_verification[:100]}")
    return {
        "user_input": user_input,
        "status": "scoped",
        "reasoning_trace": trace(state, f"\nRichiesta chiara: {decision_verification[:120]}"),
    }

# Trasforma la richiesta (eventualmente chiarita) in un brief editoriale strutturato.

def brief_node(state: dict):
    user_input = state.get("user_input", "")
    briefer = llm.with_structured_output(ResearchBrief)
    try:
        brief = briefer.invoke([SystemMessage(content=BRIEF_PROMPT.format(user_input=user_input))])
        brief_text = (f"Tema: {brief.refined_topic}\nTaglio: {brief.angle}\nNote: {brief.notes}")
        print(f"\nBrief generato. Tema: {brief.refined_topic}")
    except Exception as e:
        brief_text = ""
        print(f"\nBrief non generato ({e}): procedo con la richiesta grezza.")

    return {
        "research_brief": brief_text,
        "status": "briefed",
        "reasoning_trace": trace(state, "\nBrief editoriale strutturato generato."),
    }


# Fase 1: KG Topic Suggestion e gap analysis
def kg_context_node(state: dict):
    msgs = state.get("messages", [])
    user_input = state.get("user_input") or (msgs[-1].content if msgs else "")
    overview = kg_topics_overview()
    print("\nPanoramica di copertura recuperata dal Knowledge Graph.")

    # Recupero trend SOLO per le richieste di SUGGERIMENTO (non per la scrittura di post).
    # Quando l'utente chiede di scrivere un post su qualcosa di preciso i trend non servono al planner
    # Quando chiede "suggeriscimi argomenti" i trend sono essenziali: permettono al planner
    # di proporre temi basati sulle notizie reali invece che dalla sua conoscenza interna.
    trends = ""
    if not wants_post(user_input):
        trend_tool = tools_by_name.get("fetch_automotive_trends")
        if trend_tool:
            try:
                print("\nRecupero notizie fresche dal feed RSS")
                trends_result = str(trend_tool.invoke({"query": "novità automotive"}))
                if trends_result and "errore" not in trends_result.lower()[:50]:
                    trends = trends_result
            except Exception as e:
                print(f"\nFeed RSS non disponibile: {e}")

    return {
        "user_input": user_input,
        "kg_summary": overview,
        "trends_summary": trends,
        "status": "kg_context_loaded",
        "reasoning_trace": trace(state, "\nKG interrogato per gap/ripetizioni."),
    }


# Fase 2: Planning strutturato
# Il planner trasforma il brief in un piano strutturato di argomenti.
# Costringo il modello a rispondere con una struttura precisa che è quella del Planning.
# Passo il prompt al modello e converto i risultati del modello in dizionari Pythnon.
# Prendo il primo topic come argomento principale e il resto come argomenti secondari.

def planner_node(state: dict):
    user_input = state.get("user_input", "")
    kg_overview = state.get("kg_summary", "")
    trends = state.get("trends_summary", "") or "Nessun trend disponibile."
    brief = state.get("research_brief", "") or "Nessun brief disponibile (usa la richiesta originale)."

    planner_llm = llm.with_structured_output(PlanningSchema)
    prompt = PLANNING_PROMPT.format(
        background=default_background,
        editorial_guidelines=default_editorial_guidelines,
        kg_overview=kg_overview,
        trends=trends,
        brief=brief,
        user_input=user_input,
    )
    try:
        plan = planner_llm.invoke([SystemMessage(content=prompt)])
        # Convertiamo subito in dizionari
        planned = [p.model_dump() for p in plan.planned_posts]
        reasoning = plan.reasoning
    except Exception as e:
        planned, reasoning = [], f"(Pianificazione strutturata non riuscita: {e})."

    current_topic = planned[0]["topic"] if planned else (user_input or "Argomento automotive")
    print(f"\n{len(planned)} post pianificati. Tema scelto: {current_topic}")
    return {
        "planning_info": planned,
        "current_topic": current_topic,
        "status": "planned",
        "reasoning_trace": trace(state, "\nPiano editoriale generato."),
    }


# Metodo usato quando l'utente vuole un suggerimento. Sfrutto il planner node per
# arricchire i suggerimenti di post da scrivere, insieme alle news del tool RSS.
def suggest_topics_node(state: dict):
    plan = state.get("planning_info") or []
    trends = state.get("trends_summary", "")

    if plan:
        out = ["Proposta di calendario editoriale:\n"]
        for i, p in enumerate(plan, 1):
            out.append(f"{i}. [{p['post_category']}] {p['topic']}\n   Motivazione: {p['justification']}")
        text = "\n".join(out)
    else:
        text = "Non sono riuscito a pianificare. Specifica meglio l'area tematica."

    if trends:
        text += f"\n\nNotizie estratte dal feed RSS:\n{trends}"
    return {
        "messages": [AIMessage(content=text)],
        "status": "topics_suggested",
        "reasoning_trace": trace(state, "Presentati i topic suggeriti."),
    }


# Fase 3: ReAct + Self-RAG


# Funzione iù complessa del ReAct. Il suo compito è capire i tool da usare per trovare
# le info che servono per il topic. Tira fuori dal KG le informazioni storiche e i topic correlati
# viene fatta la query expansion a partire dalle info del KG per prendere documenti locali (RAG)
# Per questo si parla di Knowldge RAG.
def research_agent_node(state: dict):

    msgs = state.get("messages", [])
    # Al primo ingresso prepariamo system prompt + kickoff di ricerca col contesto K-RAG
    if not any(isinstance(m, ToolMessage) for m in msgs) and state.get("status") == "planned":
        topic = state.get("current_topic", "")
        # Chiave canonica per interrogare il KG che deriva dal tema principale del brief
        topic_key = canonical_topic(topic) or canonical_topic(state.get("research_brief", ""))

        # Contesto estratto dal KG
        kg_ctx = kg_topic_context(topic_key)

        # Query expansion del topic. Prendiamo i topic correlati dal Knowledge Graph e li accodiamo alla query
        # per espanderla.
        related = kg_related_topics(topic_key)
        expanded_query = topic
        if related:
            expanded_query = f"{topic} {' '.join(related)}"
            print(f"\nQuery espansa col KG: +{len(related)} topic correlati.")

        # Recuperiamo i documenti tramite RAG con la query espansa. I documenti locali sono in italiano.
        local_docs = retrieve_local(expanded_query)
        has_local = (local_docs
                     and "nessun" not in local_docs.lower()[:30]
                     and "errore" not in local_docs.lower()[:30]
                     and "non esiste" not in local_docs.lower()[:40])
        if has_local:
            local_block = local_docs
            local_sources = [f"[retrieve_local_documents] {local_docs[:600]}"]
            # Estraiamo i nomi dei file di origine per la trace (formato "- file (distanza)")
            doc_refs = re.findall(r"\[Fonte \d+ - ([^\]]+)\]", local_docs)
            rag_observation = (
                f"recuperati {len(doc_refs)} chunk dai documenti locali: "
                f"{'; '.join(doc_refs)}" if doc_refs
                else "recuperati frammenti dai documenti locali"
            )
            print("\nDocumenti locali recuperati e iniettati nel contesto.")
        else:
            local_block = "Nessun documento locale pertinente per questo tema (procedi con la ricerca web)."
            local_sources = []
            rag_observation = "nessun documento locale pertinente trovato nel database vettoriale"
            print("\nNessun documento locale rilevante per questo topic.")

        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in react_tools])
        sys = blogger_system_prompt.format(
            tools_prompt=tools_desc,
            background=default_background,
            editorial_guidelines=default_editorial_guidelines,
        )
        # Viene costruito un kickoff che inserisce nel prompt sia il context del KG che i documenti locali.
        # Così il modello conosce già elementi storici e factual del topic prima di iniziare a usare i tool.
        kickoff = RESEARCH_KICKOFF.format(topic=topic, kg_context=kg_ctx, local_docs=local_block)
        base = [SystemMessage(content=sys), HumanMessage(content=kickoff)]
        response = llm_with_tools.invoke(base)

        # Traccia del ragionamento del ReAct esplicitata nei 3 step
        # 1. Thought (pensiero), 2. Action (azione), 3. Observation (osservazione).
        expansion_note = (f" (espansa con {len(related)} topic correlati dal KG)"
                          if related else "")
        react_steps = "\n".join([
            f"Thought: Per il tema '{topic}' consulto prima la memoria del blog (KG + documenti locali) come base di conoscenza, poi validero' col web.",
            f"Action: kg_topic_context + retrieve_local_documents (K-RAG){expansion_note}.",
            f"Observation (KG): {kg_ctx[:500]}",
            f"Observation (RAG): {rag_observation}.",
            f"Thought: {response.content or 'Procedo selezionando i tool situazionali per validare e arricchire.'}",
        ])

        return {
            "messages": [SystemMessage(content=sys), HumanMessage(content=kickoff), response],
            "kg_summary": (state.get("kg_summary") or "") + "\n[KG context]\n" + kg_ctx,
            "local_sources": local_sources,
            "status": "researching",
            "reasoning_trace": trace(state, "Uso del K-RAG:\n" + react_steps),
        }

    # Codice di protezione nel caso in cui il modello usa i tool in modo errato.
    # L'app non deve crashare, ma deve procedere alla stesura con le fonti gia' raccolte (se presenti).
    try:
        response = llm_with_tools.invoke(msgs)
    except Exception as e:
        print(f"Errore nella generazione della tool call ({e}). "
              f"Procedo alla stesura con le fonti gia' raccolte.")
        return {
            "messages": [AIMessage(content="")],
            "status": "researched",
            "reasoning_trace": trace(state, f"Errore tool call ({type(e).__name__}), "
                                            f"procedo alla stesura con le fonti disponibili."),
        }
    # Se il modello chiama un tool lo registriamo come Action
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        actions = "; ".join(f"{tc.get('name')}({tc.get('args', {})})" for tc in tool_calls)
        react_line = f"Thought: {response.content or 'Mi serve un altro tool.'}\nAction: {actions}"
    else:
        react_line = f"Thought: {response.content or 'Ho abbastanza informazioni, procedo alla stesura.'}"
    return {
        "messages": [response],
        "status": "researching",
        "reasoning_trace": trace(state, "FASE 3 - ReAct:\n" + react_line),
    }

# Metodo di ricerca mirata che viene chiamato dopo la revisione da parte dell'utente (Hitl).
# A differenza di research_agent_node, non ripercorre il K-RAG completo ne' riusa
# tutta la cronologia accumulata: da' al modello un contesto pulito e una singola
# istruzione mirata a chiamare il tool giusto per il dato richiesto. Questo evita
# che un modello 3B si perda nel contesto enorme del giro precedente.
def revision_research_node(state: dict):
    feedback = state.get("human_feedback", "") or ""
    topic = state.get("current_topic", "")

    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in react_tools])
    sys = blogger_system_prompt.format(
        tools_prompt=tools_desc,
        background=default_background,
        editorial_guidelines=default_editorial_guidelines,
    )
    # Facciamo scegliere un solo tool al modello, niente deviazioni strane.
    focused = (
        f"Stai integrando un post gia' scritto sul tema '{topic}'.\n"
        f"L'unica cosa che devi fare ORA e' raccogliere i dati per questa richiesta "
        f"dell'utente: \"{feedback}\".\n\n"
        f"Scegli UN SOLO tool adatto e chiamalo:\n"
        f"- se la richiesta e' un CONFRONTO tra due veicoli -> usa 'compare_vehicles_tool';\n"
        f"- se chiede la SCHEDA TECNICA di un modello -> usa 'fetch_vehicle_specs';\n"
        f"- se chiede dati di ATTUALITA'/mercato -> usa 'mcp_web_search'.\n"
        f"Chiama subito il tool piu' adatto con gli argomenti corretti, senza scrivere altro."
    )
    base = [SystemMessage(content=sys), HumanMessage(content=focused)]
    response = llm_with_tools.invoke(base)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        actions = "; ".join(f"{tc.get('name')}({tc.get('args', {})})" for tc in tool_calls)
        react_line = f"Thought: Integro la bozza con: {feedback}.\nAction: {actions}"
    else:
        react_line = f"Thought: Integro la bozza con: {feedback} (nessun tool selezionato)."

    return {
        # Ripartiamo da una base pulita: system + istruzione mirata + risposta del modello.
        # I messaggi precedenti restano nello stato per il drafting, ma il modello qui
        # ragiona su un contesto ridotto e focalizzato.
        "messages": [SystemMessage(content=sys), HumanMessage(content=focused), response],
        "status": "researching",
        "reasoning_trace": trace(state, "Ricerca mirata per modifica:\n" + react_line),
    }

# Metodo che viene chiamato quando il modello sta per scrivere senza aver raccolto fonti.
# Serve come guardrail per evitare che il modello inventi contenuto. Inserisce una sola chiamata
# al tool mcp_web_search basata sul tema corrente, cosi' il post avra' almeno una fonte.
# Si attiva una sola volta per post (flag forced_web_search nello stato).
def forced_search_node(state: dict):
    topic = state.get("current_topic", "") or state.get("user_input", "")
    print("\nNessuna fonte raccolta: forzo una ricerca web prima della stesura.")
    forced_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "mcp_web_search",
            "args": {"query": topic},
            "id": f"forced_{int(time.time())}",
            "type": "tool_call",
        }],
    )
    return {
        "messages": [forced_call],
        "forced_web_search": True,
        "status": "researching",
        "reasoning_trace": trace(state, "\nRicerca web forzata (nessuna fonte)."),
    }

# Metodo usato dalla logica self-RAG. Valuto che le fonti raccolte dai tool vengano giudicate
# del modello, per capire se sono inerenti al contesto (es. una ricerca web che torna risultati completamente sbagliati,ecc)
def rewrite_question_node(state: dict):
    instruction = (
        "I documenti non erano rilevanti. Riformula la ricerca con parole chiave diverse "
        "(puoi usare le entita' note dal KG) oppure usa un altro tool."
    )
    return {
        "messages": [HumanMessage(content=instruction)],
        "reasoning_trace": trace(state, "Self-RAG: query riformulata."),
    }


# FASE 4 - Drafting + HITL
# Questo metodo si occupa della stesura dell'articolo, tira fuori le informazioni dal KG, citazioni
# delle fonti (grounding valido) per evitare allucinazioni, ignorando le speculazioni del modello
# e concentrandosi solo sui dati raccolti dai tool.
def drafting_node(state: dict):
    """Stesura dell'articolo con coerenza KG e citazioni dalle fonti realmente recuperate."""
    msgs = state.get("messages", [])
    topic = state.get("current_topic", "")
    # Chiave canonica dal briefing e non dall'user_input grezzo col dialogo di chiarimento.
    topic_key = canonical_topic(topic) or canonical_topic(state.get("research_brief", ""))
    consistency = kg_topic_context(topic_key)

    # Raccogliamo tutte le fonti di grounding effettivamente recuperate durante il ReAct:
    # e nessuna fonte inventata dal modello.
    GROUNDING_TOOL_NAMES = {
        "mcp_web_search", "fetch_vehicle_specs", "compare_vehicles",
        "compare_vehicles_tool", "fetch_automotive_trends", "retrieve_local_documents",
    }
    sources = []
    for m in msgs:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") in GROUNDING_TOOL_NAMES:
            content = str(m.content)
            if (content.strip()
                    and "errore" not in content.lower()[:40]
                    and "nessun" not in content.lower()[:30]):
                # La ricerca web ora restituisce un output strutturato per-fonte (titolo +
                # URL + riassunto): le diamo piu' spazio per non tagliare gli URL da citare.
                limit = 2000 if getattr(m, "name", "") == "mcp_web_search" else 900
                sources.append(f"[{m.name}] {content[:limit]}")

    # Aggiungiamo i documenti locali recuperati deterministicamente nel K-RAG
    # (non sono ToolMessage perchè iniettati nel kickoff, ma vanno citati come fonti).
    local_sources = state.get("local_sources") or []
    all_sources = local_sources + sources

    sources_block = "\n\n".join(all_sources) if all_sources else "Nessuna fonte esterna recuperata."

    prompt = DRAFT_PROMPT.format(topic=topic, kg_consistency=consistency, sources=sources_block)
    print("\nSto scrivendo l'articolo.")
    response = drafting_llm.invoke(msgs + [HumanMessage(content=prompt)])

    # Togliamo il ragionamento ReAct dalla bozza.
    clean = strip_reasoning_preamble(response.content)

    print("\nBozza completata.")
    return {
        "messages": [response],
        "draft_content": clean,
        "sources": [s[:150] for s in all_sources],
        "status": "awaiting_human_approval",
        "reasoning_trace": trace(state, "Stesura con coerenza KG e citazioni (K-RAG)."),
    }


# Metodo che implementa lo Human-in-the-loop (HITL)
# usa l'approccio interrupt(). Ci sono 4 opzioni di risposta:
# - accept -> approva la bozza e si va all'update del KG;
# - response -> feedback testuale: si torna in stesura applicando le modifiche.
# - ignore -> scarta: si termina senza aggiornare il KG.
# - edit -> modifica la bozza: si torna in stesura applicando le modifiche.
def review_node(state: dict) -> Command[Literal["update_kg_node", "drafting_node", "research_agent", "__end__"]]:
    draft = state.get("draft_content", "(nessuna bozza)")
    revisions = state.get("revision_count") or 0

    request = {
        "action_request": {
            "action": "review_post_draft",
            "args": {"topic": state.get("current_topic", "")},
        },
        "config": {
            "allow_accept": True, "allow_respond": True,
            "allow_ignore": True, "allow_edit": False,
        },
        "description": f"# Bozza pronta per la revisione\n\n{draft}",
    }
    response = interrupt(request)
    rtype = response.get("type") if isinstance(response, dict) else None

    if rtype == "accept":
        print("\nBozza approvata, procedo con l'aggiornamento del KG.")
        instruction = (
            "L'utente ha APPROVATO l'articolo. Usa ESCLUSIVAMENTE il tool "
            f"'{KG_UPDATE_TOOL_NAME}' per salvare nel KG: topic, post_title, category, "
            "sources, claims (3-4) e related_topics, estratti dalla bozza approvata."
        )
        return Command(
            goto="update_kg_node",
            update={
                "messages": [SystemMessage(content=instruction)],
                "human_feedback": "accept",
                "status": "approved",
            },
        )

    if rtype == "ignore":
        print("\nBozza scartata, termino senza aggiornare il KG.")
        return Command(goto=END, update={"human_feedback": "ignore", "status": "discarded"})

    # Qualsiasi feedback testuale mi fa tornare nella fase di drafting o in quella di research se serve.
    feedback = (
        response.get("args", "")
        if isinstance(response.get("args"), str)
        else str(response.get("args", ""))
    )
    # Se il feedback dell'utente chiede più dati allora vado nel research_node modificato
    # altrimenti torno solo al drafting con istruzioni precise.
    if modification_needs_research(feedback):
        print("\nNecessarie modifiche con nuovi dati, torno alla fase di ricerca.")
        return Command(
            goto="revision_research_node",
            update={
                "human_feedback": feedback,
                "revision_count": revisions + 1,
                # Resetto il numero massimo di ricerche web
                # Questo giro di ricerca e' completamente indipendente da quello precedente.
                "web_search_count": 0,
                "forced_web_search": False,
                "status": "revising_with_research",
            },
        )

    print("\nModifiche testuali richieste, torno alla fase di stesura.")
    instruction = (
        f"L'utente ha richiesto queste MODIFICHE testuali: '{feedback}'. "
        "Riscrivi la bozza applicandole e mantenendo le citazioni delle fonti."
    )
    return Command(
        goto="drafting_node",
        update={
            "messages": [HumanMessage(content=instruction)],
            "human_feedback": feedback,
            "revision_count": revisions + 1,
            "status": "revising",
        },
    )


# SIAMO ARRIVATI QUI A COMMENTARE, MANCANO I TOOL DOPO SEO E UTILS.


# ============================================================
# FASE 5 - Pubblicazione (solo dopo approvazione)
# ============================================================
def update_kg_node(state: dict) -> Command[Literal["__end__"]]:
    """Pubblica l'articolo approvato: genera copertina, calcola SEO e salva nel Knowledge Graph."""
    topic = state.get("current_topic") or "argomento automotive"
    plan = state.get("planning_info") or []
    category = "news"
    if plan and isinstance(plan[0], dict):
        category = plan[0].get("post_category", "news")

    draft = state.get("draft_content", "") or ""
    sources = state.get("sources") or []
    post_title = topic

    # --- 1. COPERTINA ---
    cover_path = ""
    try:
        print("[PUBLISH] Genero l'immagine di copertina (puo' richiedere qualche secondo)...")
        cover_prompt = (
            f"A stunning high-resolution photograph of the subject: '{topic}'. "
            "Professional automotive photography, photorealistic, dramatic cinematic "
            "lighting, sharp focus, ultra detailed, shot by a professional car photographer."
        )
        cover_result = generate_cover_image.invoke({"prompt": cover_prompt})
        print(f"[PUBLISH] {cover_result}")
        if "salvata" in cover_result.lower() and "'" in cover_result:
            cover_path = cover_result.split("'")[1]
    except Exception as e:
        print(f"[PUBLISH] Copertina non generata: {e}")

    # --- 2. SEO ---
    seo_score = None
    try:
        _stop = {
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
            "in", "con", "su", "per", "tra", "fra", "e", "come", "guida", "completa",
        }
        _words = [w.strip(":,.;").lower() for w in topic.split()]
        _keyword = next((w for w in _words if w and w not in _stop and len(w) > 2), topic)
        seo_report = analyze_seo_and_readability.invoke({"text": draft, "target_keyword": _keyword})
        print(f"[PUBLISH] {seo_report}")
        import textstat
        textstat.set_lang("it")
        seo_score = round(float(textstat.gulpease_index(draft)), 1)
    except Exception as e:
        print(f"[PUBLISH] Analisi SEO non riuscita: {e}")

    # --- 3. ESTRAZIONE CONOSCENZA (key claims + related topics) dal post approvato ---
    # Riempie i campi 'claims' e 'relationships' del KG richiesti dalle specifiche.
    claims, related = [], []
    try:
        extractor = llm.with_structured_output(KGExtraction)
        extraction = extractor.invoke([
            SystemMessage(content=KG_EXTRACTION_PROMPT),
            HumanMessage(content=f"Titolo: {post_title}\n\nArticolo:\n{draft[:4000]}"),
        ])
        claims = [c.strip() for c in (extraction.key_claims or []) if c and c.strip()]
        related = [r.strip().lower() for r in (extraction.related_topics or []) if r and r.strip()]
        print(f"[KG] Estratti {len(claims)} claim e {len(related)} topic correlati dal post.")
    except Exception as e:
        print(f"[KG] Estrazione claim/related non riuscita ({e}): salvo senza arricchimento.")

    # --- 4. SALVATAGGIO nel Knowledge Graph ---
    # TOPIC CANONICO deterministico: chiave breve e normalizzata (marca+modello/soggetto),
    # derivata dal TEMA PULITO (current_topic dal planner), NON dall'user_input grezzo che puo'
    # contenere il dialogo di chiarimento. Garantisce che post sullo STESSO soggetto aggancino
    # lo stesso nodo Topic -> la gap-analysis riconosce i doppioni.
    # Il TITOLO resta la stringa editoriale lunga (titolo dell'articolo).
    canon = canonical_topic(topic) or canonical_topic(state.get("research_brief", ""))
    print(f"[KG] Salvataggio del post '{post_title}' (topic canonico: '{canon}') nel Knowledge Graph...")
    result = update_kg_data(
        topic=canon,
        post_title=post_title,
        category=category,
        sources=sources,
        claims=claims,
        related_topics=related,
        content=draft,
        seo_score=seo_score,
        cover_image=cover_path,
    )
    print(f"[KG] {result}")

    return Command(goto=END, update={
        "status": "completed",
        "reasoning_trace": trace(state, "FASE 5 - Post pubblicato (copertina + SEO + salvataggio KG)."),
    })


# ============================================================
# NODO TOOL RESILIENTE
# ============================================================
def resilient_tool_node(state: dict):
    """
    Sostituisce il ToolNode standard: se il modello chiama un tool inesistente o il
    tool solleva un'eccezione, NON fa crashare il grafo. Restituisce invece un
    ToolMessage di errore come "Observation", cosi' l'agente ReAct puo' correggersi.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    outputs = []
    raw_notes_collected = []
    available = ", ".join(tools_by_name.keys())

    # Tool il cui output grezzo vale la pena conservare come "raw note" (grounding).
    # think_tool e i tool di sola lettura KG NON sono fonti, quindi esclusi.
    _GROUNDING_TOOL_NAMES = {
        "mcp_web_search", "fetch_vehicle_specs", "compare_vehicles",
        "compare_vehicles_tool", "fetch_automotive_trends", "retrieve_local_documents",
    }

    # LIMITE RICERCHE WEB: strategia "1 ricerca ricca"
    MAX_WEB_SEARCHES = 2
    # Contatore resettabile dallo stato (NON ricontiamo i messaggi storici: dopo una
    # modifica con ricerca il contatore viene azzerato in review_node, dando al nuovo
    # giro il proprio budget). Fallback: se assente, lo deriviamo dai messaggi correnti.
    web_done = state.get("web_search_count")
    if web_done is None:
        web_done = sum(
            1 for m in state.get("messages", [])
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "mcp_web_search"
        )

    for call in tool_calls:
        name = call.get("name")
        call_id = call.get("id")
        args = call.get("args", {}) or {}
        tool = tools_by_name.get(name)

        if config.debug:
            print(f"\n[DIAG] tool_call -> nome='{name}' | args={args}")

        # NORMALIZZAZIONE DIFENSIVA degli argomenti
        args = normalize_tool_args(name, args)

        # ARRICCHIMENTO SPECIFICO per fetch_vehicle_specs: se car_model non contiene
        # uno spazio (es. "TRK502X" invece di "Benelli TRK502X"), il modello ha omesso
        # il brand. Proviamo a recuperarlo dal current_topic, che di solito contiene
        # il nome completo (es. "Recensione Tecnica della TRK502X ... Benelli").
        if name == "fetch_vehicle_specs":
            car_model = args.get("car_model", "")
            if car_model and " " not in car_model.strip():
                topic = state.get("current_topic", "")
                if topic and car_model.lower() in topic.lower():
                    # Parole comuni italiane da escludere (titoli di post, articoli, preposizioni)
                    _skip = {
                        "recensione", "tecnica", "della", "del", "dello", "delle", "degli",
                        "dei", "analisi", "completa", "completo", "modello", "post", "sulla",
                        "sul", "sullo", "sulle", "confronto", "guida", "pratica", "pratico",
                        "nuova", "nuovo", "nuovi", "nuove", "dettagliata", "dettagliato",
                        "un", "una", "il", "la", "le", "li", "lo", "gli", "per", "con",
                        "tra", "fra", "come", "cosa", "chi", "perche", "panoramica",
                        "approfondimento", "storia", "caratteristiche", "specifiche",
                    }
                    # Raccogliamo TUTTI i candidati brand (no apostrofi, no parole comuni,
                    # maiuscola iniziale, non il codice modello stesso)
                    candidates = []
                    for word in topic.split():
                        clean = word.strip(":,;.()\"'")
                        if (clean
                                and "'" not in clean
                                and clean.lower() not in _skip
                                and clean.lower() != car_model.lower()
                                and clean[0].isupper()
                                and len(clean) > 2):
                            candidates.append(clean)
                    if candidates:
                        # Scegliamo il candidato PIU' VICINO al codice modello nel topic
                        # (il brand di solito appare adiacente al modello)
                        model_pos = topic.lower().find(car_model.lower())
                        best = min(candidates,
                                   key=lambda c: abs(topic.lower().find(c.lower()) - model_pos))
                        enriched = f"{best} {car_model}"
                        print(f"[Tool] car_model arricchito: '{car_model}' -> '{enriched}' (brand dal topic).")
                        args = {**args, "car_model": enriched}

        # Applica il tetto alle ricerche web
        if name == "mcp_web_search" and web_done >= MAX_WEB_SEARCHES:
            content = (
                f"Limite di {MAX_WEB_SEARCHES} ricerche web raggiunto per questo post. "
                "Usa le informazioni gia' raccolte (fonti locali e web) per scrivere la bozza, "
                "senza altre ricerche."
            )
            print(f"[Tool] Ricerca web SALTATA (limite {MAX_WEB_SEARCHES} raggiunto).")
            outputs.append(ToolMessage(content=content, name=name, tool_call_id=call_id))
            continue

        if tool is None:
            content = (
                f"ERRORE: il tool '{name}' non esiste. "
                f"Usa SOLO uno di questi tool, con il nome ESATTO: {available}."
            )
            print(f"[Tool] Nome tool inesistente '{name}': restituisco errore recuperabile.")
        else:
            try:
                content = str(tool.invoke(args))
                if name == "mcp_web_search":
                    web_done += 1
            except Exception as e:
                content = f"ERRORE durante l'esecuzione del tool '{name}': {e}. Riprova o usa un altro tool."
                print(f"[Tool] Eccezione nel tool '{name}': {e}")

        msg = ToolMessage(content=content, name=name or "unknown", tool_call_id=call_id)
        outputs.append(msg)

        # RAW NOTES (ispirato al notebook 2): conserviamo l'osservazione grezza dei
        # tool di grounding, oltre al riassunto, per averne i dettagli al drafting.
        if name in _GROUNDING_TOOL_NAMES and content.strip() and "errore" not in content.lower()[:40]:
            raw_notes_collected.append(f"[{name}] {content}")

    new_state = {"messages": outputs, "web_search_count": web_done}
    if raw_notes_collected:
        # operator.add non e' impostato sul campo: accodiamo manualmente al cumulato
        existing = state.get("raw_notes") or []
        new_state["raw_notes"] = existing + raw_notes_collected
    return new_state
