"""
Funzioni nodo del grafo LangGraph — le 5 fasi dell'agente blogger.

Ogni funzione corrisponde a un nodo nel grafo e implementa un passo del flusso:
  FASE 1 - kg_context_node:      KG interrogato per gap analysis (Topic Suggestion)
  FASE 2 - planner_node:         Planning strutturato (PlanningSchema)
           suggest_topics_node:   Presentazione suggerimenti (senza stesura)
  FASE 3 - research_agent_node:  Ciclo ReAct + Self-RAG
           forced_search_node:    Guardrail verifica fonti
           rewrite_question_node: Riformulazione query (Self-RAG)
           resilient_tool_node:   Esecuzione tool con gestione errori
  FASE 4 - drafting_node:        Stesura con coerenza KG e citazioni (K-RAG)
           review_node:           Human-in-the-loop (interrupt/Command)
  FASE 5 - update_kg_node:       Pubblicazione (copertina + SEO + salvataggio KG)
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
# Funzioni pure del KG: chiamate in modo DETERMINISTICO nelle fasi obbligatorie
from knowledge_graph.queries import kg_topics_overview, kg_topic_context, kg_related_topics
from knowledge_graph.updater import update_kg_data
# RAG: retrieval deterministico per il K-RAG (both KG and RAG)
from rag.retriever import retrieve_local
# Tool di arricchimento, eseguiti deterministicamente alla pubblicazione (post-approvazione)
from tools.image_tool import generate_cover_image
from tools.seo_tool import analyze_seo_and_readability

config = Configuration()
tools = get_all_tools()                  # tutti i tool (per lookup nel resilient_tool_node)
react_tools = get_react_tools()          # solo i 5 tool situazionali (per il bind_tools)
llm_with_tools = llm.bind_tools(react_tools)
tools_by_name = {t.name: t for t in tools}

KG_UPDATE_TOOL_NAME = "update_knowledge_graph"

# Limite di giri di chiarimento, per evitare che l'agente continui a chiedere all'infinito
MAX_CLARIFICATIONS = 2


# ============================================================
# FASE 0 - SCOPING: clarification + brief
# ============================================================
def clarification_node(state: dict):
    """
    Valuta con structured output se la richiesta dell'utente e' chiara o vaga.
    Se vaga (e non abbiamo gia' chiesto troppe volte), si ferma con interrupt() e
    chiede un chiarimento all'utente (Human-in-the-loop). Se chiara, prosegue al brief.
    """
    msgs = state.get("messages", [])
    user_input = state.get("user_input") or (msgs[-1].content if msgs else "")
    clarif_count = state.get("clarification_count") or 0

    # Se abbiamo gia' raggiunto il limite di chiarimenti, procediamo comunque (no loop infinito)
    if clarif_count >= MAX_CLARIFICATIONS:
        print("[FASE 0 - SCOPING] Limite chiarimenti raggiunto: procedo con quanto ho.")
        return {"user_input": user_input, "status": "scoped"}

    # RETE DI SICUREZZA DETERMINISTICA: alcune richieste sono palesemente generiche
    # ("scrivimi qualcosa", "fai tu", ...). Su queste forziamo SEMPRE il chiarimento,
    # senza dipendere dal giudizio del modello 3B (che oscilla troppo). Per i casi
    # sfumati lasciamo invece decidere il modello con structured output.
    # IMPORTANTE: la applichiamo solo al PRIMO passaggio. Se l'utente ha gia' fornito un
    # chiarimento (clarif_count > 0), NON la rivalutiamo (altrimenti la richiesta originale
    # generica la farebbe riscattare, stampando messaggi fuorvianti): lasciamo decidere il
    # modello sulla richiesta ormai arricchita.
    forced_vague = is_clearly_vague(user_input) if clarif_count == 0 else False

    if forced_vague:
        decision_need = True
        decision_question = (
            "La tua richiesta e' un po' generica. Su cosa ti piacerebbe scrivere? "
            "Per esempio: la recensione di un modello specifico (auto o moto), una guida "
            "tecnica (es. manutenzione, ADAS, batterie), oppure un confronto tra due veicoli?"
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
            # Se lo structured output fallisce, non blocchiamo il flusso: procediamo
            print(f"[FASE 0 - SCOPING] Clarification non riuscita ({e}): procedo senza chiarimenti.")
            return {"user_input": user_input, "status": "scoped"}

    if decision_need:
        # Interrupt HITL: stesso meccanismo della review, distinto dal campo 'action'
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

        # A questo punto l'interrupt si e' risolto: logghiamo la decisione UNA volta sola
        # (il codice sopra l'interrupt si riesegue alla ripresa, qui sotto no).
        reason_lbl = "regola deterministica" if decision_reason == "deterministica" else "valutazione del modello"
        print(f"[FASE 0 - SCOPING] Richiesta giudicata vaga ({reason_lbl}): chiesto chiarimento (HITL).")

        if rtype == "ignore":
            # L'utente non vuole chiarire: procediamo con la richiesta originale
            print("[FASE 0 - SCOPING] Nessun chiarimento fornito: procedo con la richiesta originale.")
            return {"user_input": user_input, "status": "scoped"}

        # L'utente ha risposto: arricchiamo lo user_input col chiarimento e ri-valutiamo
        extra = (answer.get("args", "") if isinstance(answer.get("args"), str)
                 else str(answer.get("args", "")))
        enriched = f"{user_input}\nChiarimento dell'utente: {extra}"
        print("[FASE 0 - SCOPING] Chiarimento ricevuto, rivaluto la richiesta.")
        return {
            "user_input": enriched,
            "clarification_count": clarif_count + 1,
            "status": "clarifying",
            "reasoning_trace": trace(state, f"FASE 0 - Chiarimento richiesto e ricevuto: {extra[:120]}"),
        }

    # Richiesta chiara: registriamo la verifica e proseguiamo
    print(f"[FASE 0 - SCOPING] Richiesta chiara: {decision_verification[:100]}")
    return {
        "user_input": user_input,
        "status": "scoped",
        "reasoning_trace": trace(state, f"FASE 0 - Richiesta chiara: {decision_verification[:120]}"),
    }


def brief_node(state: dict):
    """
    Trasforma la richiesta (eventualmente chiarita) in un BRIEF editoriale strutturato.
    """
    user_input = state.get("user_input", "")
    briefer = llm.with_structured_output(ResearchBrief)
    try:
        brief = briefer.invoke([SystemMessage(content=BRIEF_PROMPT.format(user_input=user_input))])
        brief_text = (f"Tema: {brief.refined_topic}\nTaglio: {brief.angle}\nNote: {brief.notes}")
        print(f"[FASE 0 - BRIEF] Brief generato. Tema: {brief.refined_topic}")
    except Exception as e:
        brief_text = ""
        print(f"[FASE 0 - BRIEF] Brief non generato ({e}): procedo con la richiesta grezza.")

    return {
        "research_brief": brief_text,
        "status": "briefed",
        "reasoning_trace": trace(state, "FASE 0 - Brief editoriale strutturato generato."),
    }


# ============================================================
# FASE 1 - KG: Topic suggestion (gap analysis)
# ============================================================
def kg_context_node(state: dict):
    """Interroga il KG (deterministico). Per le richieste di suggerimento, recupera anche i trend RSS."""
    msgs = state.get("messages", [])
    user_input = state.get("user_input") or (msgs[-1].content if msgs else "")
    overview = kg_topics_overview()
    print("\n[FASE 1 - KG] Panoramica di copertura recuperata dal Knowledge Graph.")

    # Recupero trend SOLO per le richieste di SUGGERIMENTO (non per la scrittura di post).
    # Quando l'utente chiede "scrivi un post sulla TRK502X" i trend non servono al planner
    # (il primo topic DEVE essere quello richiesto, per la regola di priorita').
    # Quando chiede "suggeriscimi argomenti" i trend sono essenziali: permettono al planner
    # di proporre temi basati sulle notizie reali invece che dalla sua conoscenza interna.
    trends = ""
    if not wants_post(user_input):
        trend_tool = tools_by_name.get("fetch_automotive_trends")
        if trend_tool:
            try:
                print("[FASE 1 - RSS] Recupero notizie fresche dal feed RSS...")
                trends_result = str(trend_tool.invoke({"query": "novità automotive"}))
                if trends_result and "errore" not in trends_result.lower()[:50]:
                    trends = trends_result
            except Exception as e:
                print(f"[FASE 1 - RSS] Feed RSS non disponibile: {e}")

    return {
        "user_input": user_input,
        "kg_summary": overview,
        "trends_summary": trends,
        "status": "kg_context_loaded",
        "reasoning_trace": trace(state, "FASE 1 - KG interrogato per gap/ripetizioni."),
    }


# ============================================================
# FASE 2 - Planning strutturato
# ============================================================
def planner_node(state: dict):
    """Genera una sequenza di post giustificata e KG-aware; popola planning_info."""
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
        # Convertiamo subito in dict serializzabili (evita il warning PostPlan del checkpointer)
        planned = [p.model_dump() for p in plan.planned_posts]
        reasoning = plan.reasoning
    except Exception as e:
        planned, reasoning = [], f"(Pianificazione strutturata non riuscita: {e})."

    current_topic = planned[0]["topic"] if planned else (user_input or "Argomento automotive")
    print(f"\n[FASE 2 - PLANNING] {len(planned)} post pianificati. Tema scelto: {current_topic}")
    return {
        "planning_info": planned,
        "current_topic": current_topic,
        "status": "planned",
        "reasoning_trace": trace(state, f"FASE 2 - Piano editoriale (KG-aware). {reasoning}"),
    }


def suggest_topics_node(state: dict):
    """Modalita' suggerimento: presenta il piano arricchito con trend attuali.

    I trend sono gia' stati recuperati in kg_context_node e passati al planner,
    quindi i suggerimenti sono GIA' informati dalle notizie reali. Qui li
    appendiamo alla risposta per dare visibilita' anche all'utente.
    """
    plan = state.get("planning_info") or []
    trends = state.get("trends_summary", "")

    if plan:
        out = ["Proposta di calendario editoriale (gap-aware):\n"]
        for i, p in enumerate(plan, 1):
            out.append(f"{i}. [{p['post_category']}] {p['topic']}\n   Motivazione: {p['justification']}")
        text = "\n".join(out)
    else:
        text = "Non sono riuscito a pianificare. Specifica meglio l'area tematica."

    if trends:
        text += f"\n\nNotizie fresche dalle testate automotive (feed RSS):\n{trends}"
    return {
        "messages": [AIMessage(content=text)],
        "status": "topics_suggested",
        "reasoning_trace": trace(state, "Presentati i topic suggeriti."),
    }


# ============================================================
# FASE 3 - ReAct + Self-RAG
# ============================================================
def research_agent_node(state: dict):
    """ReAct: ragiona e seleziona dinamicamente i tool per il tema scelto.

    Al primo ingresso esegue il K-RAG in modo DETERMINISTICO:
    - recupera il contesto dal Knowledge Graph (coerenza, cross-link);
    - usa i topic correlati del KG per ESPANDERE la query di retrieval (query expansion);
    - recupera i documenti locali (RAG) con la query espansa.
    Entrambe le fonti (KG + RAG) vengono iniettate nel contesto, soddisfacendo il
    requisito "use of both structured knowledge (KG) and unstructured documents (RAG)".
    Il modello sceglie poi liberamente, via ReAct, i tool SITUAZIONALI (web, specs, ecc.).
    """
    msgs = state.get("messages", [])
    # Al primo ingresso prepariamo system prompt + kickoff di ricerca col contesto K-RAG
    if not any(isinstance(m, ToolMessage) for m in msgs) and state.get("status") == "planned":
        topic = state.get("current_topic", "")
        # Chiave canonica per interrogare il KG: derivata dal TEMA PULITO (current_topic dal
        # planner), NON dall'user_input grezzo che puo' contenere il dialogo di chiarimento.
        topic_key = canonical_topic(topic) or canonical_topic(state.get("research_brief", ""))
        kg_ctx = kg_topic_context(topic_key)

        # --- QUERY EXPANSION dal KG ---
        # Prendiamo i topic correlati dal Knowledge Graph e li accodiamo alla query
        # di retrieval, per espanderla (requisito: "use the KG to expand/refine queries").
        related = kg_related_topics(topic_key)
        expanded_query = topic
        if related:
            expanded_query = f"{topic} {' '.join(related)}"
            print(f"[FASE 3 - K-RAG] Query espansa col KG: +{len(related)} topic correlati.")

        # --- RAG DETERMINISTICO ---
        # I documenti locali sono curati e in italiano. Li recuperiamo sempre con la
        # query espansa (in italiano), senza dipendere dalla scelta del modello.
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
            print("[FASE 3 - K-RAG] Documenti locali recuperati e iniettati nel contesto.")
        else:
            local_block = "Nessun documento locale pertinente per questo tema (procedi con la ricerca web)."
            local_sources = []
            rag_observation = "nessun documento locale pertinente trovato nel database vettoriale"
            print("[FASE 3 - K-RAG] Nessun documento locale rilevante per questo topic.")

        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in react_tools])
        sys = blogger_system_prompt.format(
            tools_prompt=tools_desc,
            background=default_background,
            editorial_guidelines=default_editorial_guidelines,
        )
        # Il kickoff riceve KG context + documenti locali gia' recuperati (K-RAG completo)
        kickoff = RESEARCH_KICKOFF.format(topic=topic, kg_context=kg_ctx, local_docs=local_block)
        base = [SystemMessage(content=sys), HumanMessage(content=kickoff)]
        response = llm_with_tools.invoke(base)

        # --- REASONING TRACE in formato ReAct (Thought -> Action -> Observation) ---
        # Documenta esplicitamente i passi K-RAG deterministici come richiesto dalle
        # specifiche ("explicit reasoning steps", "justification of tool usage").
        expansion_note = (f" (espansa con {len(related)} topic correlati dal KG)"
                          if related else "")
        react_steps = "\n".join([
            f"Thought: Per il tema '{topic}' consulto prima la memoria del blog (KG + documenti locali) come base di conoscenza, poi validero' col web.",
            f"Action: kg_topic_context + retrieve_local_documents (K-RAG deterministico){expansion_note}.",
            f"Observation (KG): {kg_ctx[:200]}",
            f"Observation (RAG): {rag_observation}.",
            f"Thought: {response.content or 'Procedo selezionando i tool situazionali per validare e arricchire.'}",
        ])

        return {
            "messages": [SystemMessage(content=sys), HumanMessage(content=kickoff), response],
            "kg_summary": (state.get("kg_summary") or "") + "\n[KG context]\n" + kg_ctx,
            "local_sources": local_sources,
            "status": "researching",
            "reasoning_trace": trace(state, "FASE 3 - K-RAG (KG+RAG deterministici):\n" + react_steps),
        }

    # Protezione: la generazione della tool call puo' fallire (es. il modello costruisce
    # argomenti che violano lo schema di un tool -> errore di validazione dentro invoke).
    # Non deve far cadere l'intera app: in caso di errore, procediamo verso la stesura con
    # le fonti gia' raccolte (degradazione controllata invece di crash).
    try:
        response = llm_with_tools.invoke(msgs)
    except Exception as e:
        print(f"[Research] Errore nella generazione della tool call ({e}). "
              f"Procedo alla stesura con le fonti gia' raccolte.")
        return {
            "messages": [AIMessage(content="")],
            "status": "researched",
            "reasoning_trace": trace(state, f"FASE 3 - ReAct: errore tool call ({type(e).__name__}), "
                                            f"procedo alla stesura con le fonti disponibili."),
        }
    # Trace ReAct per i passi successivi: se il modello chiama un tool lo registriamo come Action
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


def revision_research_node(state: dict):
    """
    Ricerca MIRATA per una richiesta di modifica che necessita di nuovi dati
    (es. "aggiungi un confronto con l'Audi RS5"), instradata qui dall'HITL.

    A differenza di research_agent_node, NON ripercorre il K-RAG completo ne' riusa
    tutta la cronologia accumulata: da' al modello un contesto PULITO e una singola
    istruzione mirata a chiamare il tool giusto per il dato richiesto. Questo evita
    che un modello 3B si perda nel contesto enorme del giro precedente (problema
    osservato: il compare_vehicles non veniva chiamato dopo una modifica).
    """
    feedback = state.get("human_feedback", "") or ""
    topic = state.get("current_topic", "")

    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in react_tools])
    sys = blogger_system_prompt.format(
        tools_prompt=tools_desc,
        background=default_background,
        editorial_guidelines=default_editorial_guidelines,
    )
    # Istruzione mirata: una sola azione, un solo obiettivo. Niente cronologia vecchia.
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
        "reasoning_trace": trace(state, "FASE 3bis - Ricerca mirata per modifica:\n" + react_line),
    }


def forced_search_node(state: dict):
    """
    GUARDRAIL: forza UNA ricerca web quando il modello sta per scrivere senza aver
    raccolto alcuna fonte (eviterebbe di "inventare" il contenuto). Inietta una chiamata
    al tool mcp_web_search basata sul tema corrente, cosi' il post avra' almeno una fonte.
    Si attiva una sola volta per post (flag forced_web_search nello stato).
    """
    topic = state.get("current_topic", "") or state.get("user_input", "")
    print("[Guardrail] Nessuna fonte raccolta: forzo una ricerca web prima della stesura.")
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
        "reasoning_trace": trace(state, "FASE 3 - Guardrail: ricerca web forzata (nessuna fonte)."),
    }


def rewrite_question_node(state: dict):
    """Self-RAG: istruzione per riformulare la query dopo fonti non rilevanti."""
    instruction = (
        "I documenti non erano rilevanti. Riformula la ricerca con parole chiave diverse "
        "(puoi usare le entita' note dal KG) oppure usa un altro tool."
    )
    return {
        "messages": [HumanMessage(content=instruction)],
        "reasoning_trace": trace(state, "Self-RAG: query riformulata."),
    }


# ============================================================
# FASE 4 - Drafting (KG coerenza + citazioni K-RAG)
# ============================================================
def drafting_node(state: dict):
    """Stesura dell'articolo con coerenza KG e citazioni dalle fonti realmente recuperate."""
    msgs = state.get("messages", [])
    topic = state.get("current_topic", "")
    # Chiave canonica dal TEMA PULITO (non dall'user_input grezzo col dialogo di chiarimento).
    topic_key = canonical_topic(topic) or canonical_topic(state.get("research_brief", ""))
    consistency = kg_topic_context(topic_key)

    # Raccogliamo TUTTE le fonti di grounding effettivamente recuperate durante il ReAct:
    # web search, schede tecniche (specs), confronti, trend. Sono le UNICHE fonti che il
    # modello puo' citare: in DRAFT_PROMPT gli imponiamo di non inventarne altre.
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
                limit = 1500 if getattr(m, "name", "") == "mcp_web_search" else 600
                sources.append(f"[{m.name}] {content[:limit]}")

    # Aggiungiamo i documenti locali recuperati deterministicamente nel K-RAG
    # (non sono ToolMessage perche' iniettati nel kickoff, ma vanno citati come fonti).
    local_sources = state.get("local_sources") or []
    all_sources = local_sources + sources

    sources_block = "\n\n".join(all_sources) if all_sources else "Nessuna fonte esterna recuperata."

    prompt = DRAFT_PROMPT.format(topic=topic, kg_consistency=consistency, sources=sources_block)
    print("\n[FASE 4 - DRAFT] Sto scrivendo l'articolo (puo' richiedere qualche minuto)...")
    response = drafting_llm.invoke(msgs + [HumanMessage(content=prompt)])

    # PULIZIA DIFENSIVA: alcuni modelli locali antepongono il ragionamento ReAct
    clean = strip_reasoning_preamble(response.content)

    print("[FASE 4 - DRAFT] Bozza completata.")
    return {
        "messages": [response],
        "draft_content": clean,
        "sources": [s[:150] for s in all_sources],
        "status": "awaiting_human_approval",
        "reasoning_trace": trace(state, "FASE 4 - Stesura con coerenza KG e citazioni (K-RAG)."),
    }


# ============================================================
# FASE 4 (HITL) - pattern interrupt()/Command del tutorial
# ============================================================
def review_node(state: dict) -> Command[Literal["update_kg_node", "drafting_node", "research_agent", "__end__"]]:
    """
    Human-in-the-loop: presenta la bozza con interrupt() e gestisce la risposta.
    Tipi di risposta (come nel tutorial: accept/edit/ignore/response):
      - 'accept'   -> approva: si va all'update del KG.
      - 'response' -> feedback testuale: si torna in stesura applicando le modifiche.
      - 'ignore'   -> scarta: si termina senza aggiornare il KG.
    """
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
        print("\n[HITL] Bozza APPROVATA -> aggiornamento KG.")
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
        print("\n[HITL] Bozza SCARTATA -> fine senza aggiornare il KG.")
        return Command(goto=END, update={"human_feedback": "ignore", "status": "discarded"})

    # 'response' (o qualsiasi feedback testuale) -> modifiche e ritorno in stesura
    feedback = (
        response.get("args", "")
        if isinstance(response.get("args"), str)
        else str(response.get("args", ""))
    )
    # SMART ROUTING della modifica (HITL piu' completo):
    # - se il feedback richiede NUOVI DATI (es. "aggiungi un confronto con l'Audi RS3"),
    #   torniamo al research_agent, che puo' chiamare i tool (compare, specs, web) e poi
    #   ri-passa dal drafting con le nuove informazioni;
    # - se e' una modifica puramente TESTUALE (accorcia, cambia tono...), restiamo sul
    #   drafting_node, piu' veloce, come prima.
    if modification_needs_research(feedback):
        print("\n[HITL] MODIFICHE con NUOVI DATI -> ricerca mirata (revision_research_node).")
        return Command(
            goto="revision_research_node",
            update={
                "human_feedback": feedback,
                "revision_count": revisions + 1,
                # Reset del contatore ricerche web: il nuovo giro di integrazione ha
                # diritto al proprio budget di ricerche, indipendente dal giro precedente.
                "web_search_count": 0,
                "forced_web_search": False,
                "status": "revising_with_research",
            },
        )

    print("\n[HITL] MODIFICHE testuali -> ritorno in stesura (drafting).")
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
