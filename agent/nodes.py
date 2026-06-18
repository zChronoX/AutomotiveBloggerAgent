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
import json
import time
from typing import Literal

from langgraph.types import interrupt, Command
from langgraph.graph import END

#Formato standard con cui "dialogare" con gli LLM. ToolMessage è essenziale per registrare gli strumenti usati.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage

from .llm import llm, drafting_llm
from .state import (
    PlanningSchema, PostPlan, ClarifyWithUser, ResearchBrief, KGExtraction,
    EditorialDecision, ProposalAction,
)
from .helpers import (
    trace, has_collected_sources, strip_reasoning_preamble, normalize_tool_args,
    wants_post, modification_needs_research, is_clearly_vague, canonical_topic,
    extract_num_posts,
)
from config.settings import Configuration
from prompts.system_prompts import (
    blogger_system_prompt, default_background, default_editorial_guidelines,
)
from prompts.agent_prompts import (
    PLANNING_PROMPT, RESEARCH_KICKOFF, DRAFT_PROMPT,
    CLARIFICATION_PROMPT, BRIEF_PROMPT, KG_EXTRACTION_PROMPT,
    REPLAN_ONE_PROMPT, PROPOSE_MORE_PROMPT,
)
from utils import should_grade_tool
from tools.base import get_all_tools, get_react_tools
# Funzioni pure del KG chiamate in modo deterministico nelle fasi obbligatorie
from knowledge_graph.queries import (
    kg_topics_overview, kg_topic_context, kg_related_topics, kg_pending_proposals,
    kg_pending_titles_list,
    kg_recent_posts, kg_proposed_titles,
)
from knowledge_graph.updater import update_kg_data, add_proposals, remove_proposal
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
        print("Limite chiarimenti raggiunto, procedo comunque con il briefing.")
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
            print(f"Clarification non riuscita ({e}): procedo senza chiarimenti.")
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
        print(f"\nRichiesta giudicata vaga: chiesto chiarimento.")

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

    # Richiesta chiara: registriamo la verifica e proseguiamo.
    # In console non stampiamo la "motivazione" del modello: il 3B spesso non motiva ma
    # risponde alla richiesta (es. propone temi), confondendo l'utente.
    print("\nRichiesta chiara: procedo con la pianificazione.")
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

    # Recupero trend solo per le richieste di suggerimento (non per la scrittura di post).
    # Quando l'utente chiede di scrivere un post su qualcosa di preciso i trend non servono al planner
    # Quando chiede "suggeriscimi argomenti" i trend sono essenziali: permettono al planner
    # di proporre temi basati sulle notizie reali invece che dalla sua conoscenza interna.
    trends = ""
    if not wants_post(user_input):
        trend_tool = tools_by_name.get("fetch_automotive_trends")
        if trend_tool:
            try:
                print("\nRecupero notizie dal feed RSS")
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

    # Contesto per la continuità e per evitare doppioni/invenzioni:
    # - i post già pubblicati (titoli reali): se l'utente chiede un confronto/seguito con
    #   "qualcosa di già trattato", il planner deve pescare un modello REALE da qui;
    # - le proposte in sospeso: così sa cosa c'è già in backlog ed evita di riproporlo.
    published_posts = kg_recent_posts() or "(nessun post pubblicato finora)"
    pending_proposals = kg_proposed_titles() or "(nessuna proposta in sospeso)"

    # Numero di post da pianificare estratto dinamicamente dalla richiesta dell'utente
    # (es. "pianifica 4 post"). Se non specificato vale il default. Rappresenta il numero massimo
    # di proposte; quante scriverne lo decide l'utente al gate editoriale.
    max_posts = extract_num_posts(user_input)

    planner_llm = llm.with_structured_output(PlanningSchema)
    prompt = PLANNING_PROMPT.format(
        background=default_background,
        editorial_guidelines=default_editorial_guidelines,
        kg_overview=kg_overview,
        trends=trends,
        brief=brief,
        user_input=user_input,
        max_posts=max_posts,
        published_posts=published_posts,
        pending_proposals=pending_proposals,
    )
    try:
        plan = planner_llm.invoke([SystemMessage(content=prompt)])
        # Convertiamo subito in dizionari
        planned = [p.model_dump() for p in plan.planned_posts]
        reasoning = plan.reasoning
    except Exception as e:
        planned, reasoning = [], f"(Pianificazione strutturata non riuscita: {e})."

    # Il modello 3B a volte ignorava la regola
    # "esattamente N" e generava comunque 3 proposte.
    # Tagliamo qui per garantire il conteggio richiesto, senza dipendere dall'LLM.
    if max_posts and len(planned) > max_posts:
        planned = planned[:max_posts]

    current_topic = planned[0]["topic"] if planned else (user_input or "Argomento automotive")
    print(f"\n{len(planned)} post pianificati (max richiesti: {max_posts}). Tema principale: {current_topic}")
    return {
        "planning_info": planned,
        "current_topic": current_topic,
        "num_posts_requested": max_posts,
        "status": "planned",
        "reasoning_trace": trace(state, "\nPiano editoriale generato."),
    }



# Fase intermedia di planning
# Dopo il planner l'utente non parte automaticamente a scrivere: rivede le proposte e
# decide quali scrivere, quali modificare (con istruzioni), quali scartare e se vuole
# proposte nuove per tornare al numero richiesto. Le proposte vengono salvate subito
# come 'proposed' e si entra nel ciclo di scrittura un post alla volta.
# Il nodo si auto-instrada a se stesso (per ri-presentare il piano dopo le
# modifiche), a research_agent (per iniziare a scrivere) o a END (annullamento).

# Formatta la lista di proposte in modo numerato e leggibile (per la presentazione HITL).
def _format_proposals(plan: list) -> str:
    if not plan:
        return "(nessuna proposta disponibile)"
    out = []
    for i, p in enumerate(plan, 1):
        cat = p.get("post_category") or p.get("category") or "n/d"
        out.append(f"{i}. [{cat}] {p.get('topic', '')}\n   Motivazione: {p.get('justification', '')}")
    return "\n".join(out)


# Converte un PostPlan (dict) nel formato di salvataggio del backlog proposte.
# 'title' resta leggibile per la presentazione; 'topic_key' e' la chiave canonica usata
# per il match/dedup e per la rimozione quando il post viene pubblicato.
def _proposal_to_storage(p: dict) -> dict:
    title = p.get("topic", "") or "(senza titolo)"
    return {
        "title": title,
        "topic_key": canonical_topic(title),
        "category": p.get("post_category") or p.get("category") or "news",
        "justification": p.get("justification", ""),
    }


# Costruisce l'aggiornamento di stato per ripartire pulito su un nuovo post.
# Svuota il canale messaggi con rimozione per-id e azzera tutti gli
# accumulatori per-post. Imposta status='planned' così research_agent rifà il
# kickoff K-RAG da zero sul nuovo topic.
def _reset_for_new_post(state: dict, next_post: dict) -> dict:
    removals = [RemoveMessage(id=m.id) for m in state.get("messages", []) if getattr(m, "id", None)]
    topic = next_post.get("topic", "") if isinstance(next_post, dict) else ""
    return {
        "messages": removals,
        "raw_notes": [],
        "sources": [],
        "local_sources": [],
        "draft_content": "",
        "web_search_count": 0,
        "done_tool_calls": [],
        "forced_web_search": False,
        "revision_count": 0,
        "human_feedback": None,
        "current_post": next_post,
        "current_topic": topic,
        "status": "planned",
    }


# Verbi per il parsing deterministico della decisione editoriale.
# Non usiamo il modello per capire quali proposte e quale operazione
# (sui test il modellino sbagliava i numeri e ignorava il refill). Numeri e operazione si
# ricavano con regex affidabili; il modello resta usato SOLO per rigenerare il testo
# di una proposta o per generare quelle nuove, cioe' dove serve creativita'.
_ED_DROP = ("scart", "elimin", "rimuov", "cancell", "togli il", "togli la", "togli i", "togli lo", "non mi interess", "leva il", "leva la")
_ED_MODIFY = ("modific", "rendil", "rendet", "rendi ", "cambia", "trasform", "fallo", "falla", "fai un", "fai una", "allung", "accorci", "aggiungi", "invece", "confront", "recensi", "trasformal")
_ED_WRITE = ("scriv", "tieni", "tien", "va bene", "vanno bene", "approv", "ok", "conferma", "procedi", "manten", "accett", "questi", "questo")
# request_new: richiesta esplicita di nuove proposte (refill).
_ED_NEW_RE = r"(propon\w*|rimpiazz\w*|sostitu\w*|aggiung\w*\s+(un|una|altr|nuov|altre|altri|qualc))"


def _parse_editorial_decision(user_response: str, plan: list) -> EditorialDecision:
    """
    Interpreta in modo deterministico la risposta dell'utente al gate editoriale.
    Spezza la frase in clausole (separatori ; . a capo), e per ogni clausola ricava:
    i numeri delle proposte coinvolte e l'operazione dal verbo usato. Per le modifiche
    si prende solo il primo numero come bersaglio.
    """
    dec = EditorialDecision()
    text = (user_response or "").strip()
    if not text or not plan:
        return dec
    low = text.lower()
    n_plan = len(plan)

    def in_range(nums):
        return [x for x in nums if 1 <= x <= n_plan]

    # Segnale esplicito di nuove proposte (e relativo spunto, se presente).
    m_new = re.search(_ED_NEW_RE, low)
    if m_new:
        dec.request_new = True
        # Lo spunto e' la parte di frase attorno alla richiesta di novita' (clausola).
        for clause in re.split(r"[;\n.]+", text):
            if re.search(_ED_NEW_RE, clause.lower()):
                dec.new_hint = clause.strip()
                break

    # Azioni per-proposta, clausola per clausola.
    seen = set()
    for clause in re.split(r"[;\n.]+", text):
        c = clause.strip()
        if not c:
            continue
        cl = c.lower()
        nums = in_range([int(x) for x in re.findall(r"\d+", c)])

        # Una clausola di sola richiesta-nuove non e' un'azione su proposte esistenti.
        if re.search(_ED_NEW_RE, cl) and not any(v in cl for v in _ED_DROP + _ED_MODIFY):
            continue

        if any(v in cl for v in _ED_DROP):
            for x in nums:
                if x not in seen:
                    dec.actions.append(ProposalAction(index=x, action="drop")); seen.add(x)
        elif any(v in cl for v in _ED_MODIFY):
            if nums:
                x = nums[0]  # solo il primo numero e' il bersaglio della modifica
                if x not in seen:
                    dec.actions.append(ProposalAction(index=x, action="modify", instruction=c)); seen.add(x)
        elif any(v in cl for v in _ED_WRITE):
            targets = nums
            if not targets and re.search(r"tutt[ie]", cl):
                targets = list(range(1, n_plan + 1))
            # Se c'e' un verbo di scrittura senza numero e senza "tutti", ma c'e' UNA
            # sola proposta, l'utente intende chiaramente quella (es. "scrivilo", "procedi
            # pure", "puoi scriverlo"). Vale anche se le proposte sono poche: "scrivi" senza
            # numeri su un piano da 1-2 proposte = scrivi tutte quelle rimaste.
            if not targets and n_plan <= 2:
                targets = list(range(1, n_plan + 1))
            for x in targets:
                if x not in seen:
                    dec.actions.append(ProposalAction(index=x, action="write")); seen.add(x)
        elif nums:
            # Numeri "nudi" senza verbo riconosciuto: li interpreto come "scrivi questi".
            for x in nums:
                if x not in seen:
                    dec.actions.append(ProposalAction(index=x, action="write")); seen.add(x)

    # Se scrivo di procedere (seguendo quelle parole) allora scrivo il post in autoamtico
    if not dec.actions and not dec.request_new:
        if (re.search(r"tutt[ie]", low) or re.search(r"procedi|vanno bene|va bene|conferma|approv|procedi pure", low)) \
           and not any(v in low for v in _ED_DROP) and not any(v in low for v in _ED_MODIFY):
            dec.actions = [ProposalAction(index=i + 1, action="write") for i in range(n_plan)]

    return dec


# Rigenera una singola proposta applicando l'istruzione dell'utente. Le proposte che
# l'utente non chiede di modificare non passano mai di qui: restano identiche.
def _regenerate_proposal(orig: dict, instruction: str, brief: str, kg_overview: str, published_posts: str = "") -> dict:
    gen = llm.with_structured_output(PostPlan)
    prompt = REPLAN_ONE_PROMPT.format(
        topic=orig.get("topic", ""),
        category=orig.get("post_category") or orig.get("category") or "",
        justification=orig.get("justification", ""),
        brief=brief or "Nessun brief.",
        kg_overview=kg_overview or "Nessuna copertura nota.",
        published_posts=published_posts or "(nessun post pubblicato finora)",
        instruction=instruction,
    )
    return gen.invoke([SystemMessage(content=prompt)]).model_dump()


# Genera k proposte aggiuntive per il refill, evitando i temi gia' tenuti o scartati.
def _propose_more(k: int, brief: str, kg_overview: str, trends: str, exclude: list, hint: str = "", published_posts: str = "") -> list:
    gen = llm.with_structured_output(PlanningSchema)
    excl = "; ".join([e for e in exclude if e]) or "(nessuno)"
    prompt = PROPOSE_MORE_PROMPT.format(
        k=k, brief=brief or "Nessun brief.", kg_overview=kg_overview or "Nessuna copertura nota.",
        published_posts=published_posts or "(nessun post pubblicato finora)",
        trends=trends or "Nessun trend.", exclude=excl,
    )
    if hint:
        prompt += f"\n\nSpunto specifico richiesto dall'utente per le nuove proposte: {hint}"
    res = gen.invoke([SystemMessage(content=prompt)])
    return [p.model_dump() for p in res.planned_posts][:k]


def editorial_review_node(state: dict) -> Command[Literal["editorial_review_node", "research_agent", "__end__"]]:
    plan = state.get("planning_info") or []
    n = state.get("num_posts_requested") or (len(plan) or 3)

    # Presentazione (codice eseguito anche ad ogni ripresa dell'interrupt).
    shortfall = max(0, n - len(plan))
    legend = (
        "Cosa vuoi fare? Esempi:\n"
        "- \"scrivi 1 e 3\"\n"
        "- \"scrivi 1; il 2 rendilo un confronto con la BMW Serie 3; scarta il 3\"\n"
        "- \"scrivili tutti\"   - \"annulla\""
    )
    if shortfall > 0:
        legend += (
            f"\n\nHai {len(plan)} proposte attive (ne avevi chieste {n}). Puoi dirmi "
            f"\"proponi nuove\" per riportarle a {n}, oppure procedere con queste."
        )
    description = "# Proposte editoriali\n\n" + _format_proposals(plan) + "\n\n" + legend

    request = {
        "action_request": {"action": "review_editorial_plan", "args": {}},
        "config": {
            "allow_accept": False, "allow_respond": True,
            "allow_ignore": True, "allow_edit": False,
        },
        "description": description,
    }
    answer = interrupt(request)
    rtype = answer.get("type") if isinstance(answer, dict) else None

    if rtype == "ignore":
        print("\nPianificazione annullata: niente da scrivere e niente salvato.")
        return Command(goto=END, update={"status": "planning_cancelled"})

    if not plan:
        print("\nNessuna proposta disponibile: chiudo.")
        return Command(goto=END, update={"status": "planning_cancelled"})

    user_response = (answer.get("args", "") if isinstance(answer.get("args"), str)
                     else str(answer.get("args", "")))
    decision = _parse_editorial_decision(user_response, plan)

    # Validazione e smistamento delle azioni.
    # Da un'unica lista di azioni ricaviamo i tre insiemi che servono al resto del nodo.
    # In caso di indice duplicato vince l'ultima azione indicata per quella proposta.
    total = len(plan)
    action_by_idx = {}
    for a in (decision.actions or []):
        if 1 <= a.index <= total and a.action in ("write", "modify", "drop"):
            action_by_idx[a.index - 1] = (a.action, (a.instruction or "").strip())
    write_idx = sorted([i for i, (act, _) in action_by_idx.items() if act == "write"])
    drop_idx = {i for i, (act, _) in action_by_idx.items() if act == "drop"}
    # Una 'modify' senza istruzione non va bene, viene trattata come approvazione.
    modify = [(i, instr) for i, (act, instr) in action_by_idx.items() if act == "modify" and instr]
    write_idx = sorted(set(write_idx) | {i for i, (act, instr) in action_by_idx.items() if act == "modify" and not instr})
    request_new = bool(decision.request_new)
    structural = bool(modify or drop_idx or request_new)

    if not write_idx and not structural:
        print("\nNon ho capito la scelta: ripropongo il piano. (Suggerimento: indica i numeri "
              "delle proposte con scrivi/modifica/scarta. Se invece vuoi fare una richiesta "
              "diversa, digita 'annulla' e riformulala dal prompt principale.)")
        return Command(goto="editorial_review_node")

    brief = state.get("research_brief", "") or ""
    kg_overview = state.get("kg_summary", "") or ""
    trends = state.get("trends_summary", "") or "Nessun trend disponibile."
    # Titoli reali dei post gia' pubblicati: servono a modify/refill per riferirsi a
    # modelli reali (continuità) invece di inventarne (es. "Giulia Quattrosotto").
    published_posts = kg_recent_posts() or "(nessun post pubblicato finora)"

    # Cambiamenti strutturali espliciti (modifica/scarto/refill): applico e ri-presento.
    if structural:
        new_plan = list(plan)
        for i, instr in modify:
            try:
                new_plan[i] = _regenerate_proposal(new_plan[i], instr, brief, kg_overview, published_posts)
                print(f"\nProposta {i + 1} rigenerata con la modifica richiesta.")
            except Exception as e:
                print(f"\nRigenerazione proposta {i + 1} non riuscita ({e}): la lascio invariata.")

        rejected = list(state.get("rejected_topics") or [])
        for i in sorted(drop_idx):
            rejected.append(canonical_topic(plan[i].get("topic", "")))
        kept_plan = [p for j, p in enumerate(new_plan) if j not in drop_idx]

        if request_new:
            k = max(0, n - len(kept_plan))
            if k > 0:
                exclude = [p.get("topic", "") for p in kept_plan] + [t for t in rejected if t]
                try:
                    extra = _propose_more(k, brief, kg_overview, trends, exclude,
                                          hint=decision.new_hint, published_posts=published_posts)
                    kept_plan = kept_plan + extra
                    print(f"\nAggiunte {len(extra)} nuove proposte (refill).")
                except Exception as e:
                    print(f"\nRefill non riuscito ({e}).")

        return Command(goto="editorial_review_node", update={
            "planning_info": kept_plan,
            "rejected_topics": [t for t in rejected if t],
            "reasoning_trace": trace(state, "Gate editoriale: piano aggiornato (modifiche/scarti/refill)."),
        })

    # Qui approvo solo le proposte, senza fare modifiche esplicite.
    # Nel caso in cui l'utente approva la prima e la terza proposta su 3, la proposta 2 viene
    # scartata implicitamente. Il piano torna a 2, ma l'utente ne aveva chiesti 3. Quindi
    # compare l'opzione refill nel caso in cui volesse altre proposte.
    selected_idx = sorted(set(write_idx))
    if not selected_idx:
        return Command(goto="editorial_review_node")

    if set(selected_idx) != set(range(len(plan))):
        rejected = list(state.get("rejected_topics") or [])
        for j, p in enumerate(plan):
            if j not in selected_idx:
                rejected.append(canonical_topic(p.get("topic", "")))
        reduced = [plan[i] for i in selected_idx]
        print(f"\nTengo {len(reduced)} proposte; le altre le ho scartate.")
        return Command(goto="editorial_review_node", update={
            "planning_info": reduced,
            "rejected_topics": [t for t in rejected if t],
            "reasoning_trace": trace(state, "Gate editoriale: selezione ridotta, ripropongo (eventuale refill)."),
        })

    # L'utente ha selezionato tutte le proposte mostrate
    # Salvo tutte le proposte come proposal, e procedo con la prima
    selected = list(plan)
    try:
        print(f"\n{add_proposals([_proposal_to_storage(p) for p in selected])}")
    except Exception as e:
        print(f"\nSalvataggio proposte non riuscito ({e}).")

    queue = list(selected)
    first = queue.pop(0)
    reset = _reset_for_new_post(state, first)
    reset["selected_posts"] = queue
    reset["reasoning_trace"] = trace(state, f"Gate editoriale: {len(selected)} post selezionati.")
    print(f"\n{len(selected)} post selezionati. Inizio a scrivere: {first.get('topic', '')}")
    return Command(goto="research_agent", update=reset)


# Metodo usato quando l'utente vuole un suggerimento. Sfrutto il planner node per
# arricchire i suggerimenti di post da scrivere, insieme alle news del tool RSS.
# Parser deterministico della scelta al gate dei suggerimenti. L'utente puo' indicare:
# - "proposta 2" / "sospesa 1" -> proposte pendenti dal KG;
# - "calendario 1" / "piano 2" -> proposte del nuovo calendario;
# - "notizia 3" / "rss 3"      -> notizie del feed;
# - un numero nudo             -> in ordine di priorita' (pendenti > calendario > rss);
# - testo libero               -> match per sottostringa sui titoli, altrimenti tema libero.
# Restituisce il tema scelto (stringa) o None se vuoto/incomprensibile.
def _parse_suggestion_choice(text: str, pending: list, plan_topics: list, rss_titles: list):
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()

    def _pick(lst, n):
        return lst[n - 1] if 1 <= n <= len(lst) else None

    # Riferimento esplicito a una lista + numero.
    m = re.search(r"(?:propost\w*|sospes\w*|recuperabil\w*)\D{0,12}(\d+)", low)
    if m:
        return _pick(pending, int(m.group(1)))
    m = re.search(r"(?:calendario|piano|nuov\w*)\D{0,12}(\d+)", low)
    if m:
        return _pick(plan_topics, int(m.group(1)))
    m = re.search(r"(?:notizi\w*|rss|feed|trend\w*)\D{0,12}(\d+)", low)
    if m:
        return _pick(rss_titles, int(m.group(1)))

    # Numero nudo: stessa priorita' con cui le liste vengono mostrate.
    m = re.search(r"\b(\d+)\b", low)
    if m:
        n = int(m.group(1))
        return _pick(pending, n) or _pick(plan_topics, n) or _pick(rss_titles, n)

    # Match per sottostringa sui titoli (es. "quello sulla Porsche").
    # Cerco le parole significative della risposta dentro i titoli.
    words = [w for w in re.findall(r"[a-zA-Zàèéìòù0-9]{4,}", low)
             if w not in ("quello", "quella", "sulla", "sullo", "sulle", "sugli",
                          "post", "articolo", "scrivi", "scrivere", "vorrei", "facciamo")]
    if words:
        for lst in (pending, plan_topics, rss_titles):
            for title in lst:
                tl = (title or "").lower()
                if any(w in tl for w in words):
                    return title

    # Testo libero con un minimo di sostanza: lo uso come tema nuovo.
    if len(t) >= 8:
        return t
    return None


# Presenta i suggerimenti (proposte in sospeso > calendario > RSS) e poi chiede
# all'utente se vuole scrivere uno dei temi proposti (HITL "choose_suggestion").
# Se sceglie un tema, il flusso riparte dal brief con la nuova richiesta di scrittura;
# se rifiuta (invio/no), si chiude come prima lasciando i suggerimenti come risposta.
def suggest_topics_node(state: dict) -> Command[Literal["suggest_topics_node", "brief_node", "__end__"]]:
    plan = state.get("planning_info") or []
    trends = state.get("trends_summary", "")

    # Ordine di PRIORITA' richiesto: prima le proposte in sospeso (recuperabili da piani
    # precedenti), poi il nuovo calendario editoriale, infine il feed RSS come ripiego.
    parts = []

    # 1) Proposte in sospeso (proposed) -> priorita' massima.
    pending_text = kg_pending_proposals()
    pending_titles = kg_pending_titles_list()
    if pending_text:
        parts.append(pending_text + "\n")

    # 2) Nuovo calendario editoriale proposto dal planner.
    plan_topics = [p.get("topic", "") for p in plan]
    if plan:
        out = ["Proposta di calendario editoriale:\n"]
        for i, p in enumerate(plan, 1):
            out.append(f"{i}. [{p['post_category']}] {p['topic']}\n   Motivazione: {p['justification']}\n")
        parts.append("\n".join(out))
    elif not pending_text:
        # Niente di pianificato e nessuna proposta in sospeso: chiedo di precisare.
        parts.append("Non sono riuscito a pianificare. Specifica meglio l'area tematica.\n")

    # 3) Feed RSS come alternativa, solo come ripiego rispetto ai punti sopra.
    rss_titles = []
    if trends:
        rss_titles = [m.group(1).strip() for m in re.finditer(r"^\s*\d+\.\s*(.+)$", trends, re.M)]
        parts.append(
            "Se i post in sospeso e quelli del calendario editoriale non dovessero "
            "piacerti, possiamo partire da una di queste notizie fresche dal feed RSS:\n"
            + trends
        )

    text = "\n\n".join(parts) if parts else "Non ho proposte da mostrare al momento."

    # Chiedo se vuole scrivere uno dei temi proposti.
    request = {
        "action_request": {"action": "choose_suggestion", "args": {}},
        "config": {
            "allow_accept": False, "allow_respond": True,
            "allow_ignore": True, "allow_edit": False,
        },
        "description": text,
    }
    answer = interrupt(request)
    rtype = answer.get("type") if isinstance(answer, dict) else None

    if rtype != "response":
        # L'utente non vuole scrivere ora: chiudo lasciando i suggerimenti come risposta.
        return Command(goto=END, update={
            "messages": [AIMessage(content=text)],
            "status": "topics_suggested",
            "reasoning_trace": trace(state, "Presentati i topic suggeriti; l'utente non ha scelto nulla da scrivere."),
        })

    scelta = (answer.get("args", "") if isinstance(answer.get("args"), str)
              else str(answer.get("args", "")))
    tema = _parse_suggestion_choice(scelta, pending_titles, plan_topics, rss_titles)

    if not tema:
        print("\nNon ho capito quale tema vuoi: ripropongo i suggerimenti. "
              "(Indica ad esempio 'la proposta 1', 'la notizia 3', oppure descrivi il tema.)")
        return Command(goto="suggest_topics_node")

    # Tema scelto: costruisco la nuova richiesta di scrittura e riparto dal brief,
    # riusando l'intero flusso (brief -> KG -> planner -> gate editoriale ->).
    new_input = f"Scrivi un post su: {tema}"
    print(f"\nOttimo: preparo il piano per '{tema}'.")
    return Command(goto="brief_node", update={
        "user_input": new_input,
        "status": "scoped",
        "reasoning_trace": trace(state, f"L'utente ha scelto dai suggerimenti: {tema}."),
    })


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
            local_sources = [f"[retrieve_local_documents] {local_docs[:800]}"]
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
                # compare_vehicles ha un verdetto lungo (~1400 char con 4 categorie):
                # anche quello va dato intero per non troncare le ultime categorie.
                if getattr(m, "name", "") in ("mcp_web_search", "compare_vehicles", "compare_vehicles_tool"):
                    limit = 2000
                else:
                    limit = 900
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
def review_node(state: dict) -> Command[Literal["update_kg_node", "drafting_node", "revision_research_node", "next_post_node"]]:
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
        # Bozza scartata: il post NON viene pubblicato, quindi resta nel backlog come
        # proposta (non lo rimuovo) ed e' recuperabile. Vado al gate 'prossimo post'
        # invece di terminare, cosi' se restano selezionati l'utente decide se proseguire.
        print("\nBozza scartata: non aggiorno il KG. Il post resta tra le proposte.")
        return Command(goto="next_post_node", update={"human_feedback": "ignore", "status": "discarded"})

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





# Fase 5 - Aggiornamento del KG, genera copertina, calcola SEO e salva nel Knowledge Graph.

# Nodo di pubblicazione per il  KG. Viene eseguito all'approvazione del post.
# recupero lo stato del grafo, come topic, categoria, draft e fonti.
# Genero l'immagine di copertina, analisi SEO e estraggo claim correlati e argomenti correlati
# per il KG.
def update_kg_node(state: dict) -> Command[Literal["next_post_node"]]:
    topic = state.get("current_topic") or "argomento automotive"
    # La categoria viene dal post corrente col ciclo multi-post il post in scrittura
    # potrebbe non essere il primo del piano. Fallback su planning_info[0] per sicurezza.
    current = state.get("current_post") or {}
    category = current.get("post_category") or current.get("category")
    if not category:
        plan = state.get("planning_info") or []
        if plan and isinstance(plan[0], dict):
            category = plan[0].get("post_category", "news")
    category = category or "news"

    draft = state.get("draft_content", "") or ""
    sources = state.get("sources") or []
    post_title = topic

    # Tool generazione immagini
    cover_path = ""
    try:
        print("Genero l'immagine di copertina.")
        cover_prompt = (
            f"A stunning high-resolution photograph of the subject: '{topic}'. "
            "Professional automotive photography, photorealistic, dramatic cinematic "
            "lighting, sharp focus, ultra detailed, shot by a professional car photographer."
            "Don't include any text in the image."
        )
        cover_result = generate_cover_image.invoke({"prompt": cover_prompt})
        print(f"\n{cover_result}")
        if "salvata" in cover_result.lower() and "'" in cover_result:
            cover_path = cover_result.split("'")[1]
    except Exception as e:
        print(f"\nCopertina non generata: {e}")

    # Tool SEO
    seo_score = None
    try:
        _stop = {
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
            "in", "con", "su", "per", "tra", "fra", "e", "come", "guida", "completa",
        }
        _words = [w.strip(":,.;").lower() for w in topic.split()]
        _keyword = next((w for w in _words if w and w not in _stop and len(w) > 2), topic)
        seo_report = analyze_seo_and_readability.invoke({"text": draft, "target_keyword": _keyword})
        print(f"\n{seo_report}")
        import textstat
        textstat.set_lang("it")
        seo_score = round(float(textstat.gulpease_index(draft)), 1)
    except Exception as e:
        print(f"\nAnalisi SEO non riuscita: {e}")

    # Estrazione claims e related topics.
    claims, related = [], []
    try:
        extractor = llm.with_structured_output(KGExtraction)
        extraction = extractor.invoke([
            SystemMessage(content=KG_EXTRACTION_PROMPT),
            HumanMessage(content=f"Titolo: {post_title}\n\nArticolo:\n{draft[:4000]}"),
        ])
        claims = [c.strip() for c in (extraction.key_claims or []) if c and c.strip()]
        related = [r.strip().lower() for r in (extraction.related_topics or []) if r and r.strip()]
        print(f"\nEstratti {len(claims)} claim e {len(related)} topic correlati dal post.")
    except Exception as e:
        print(f"\nEstrazione claim/related non riuscita ({e}): salvo senza arricchimento.")

    # Normalizzo la chiave come sopra, prendendola dal briefing e non dall'input
    # dell'utente che può contenere la conversazione con i chiarimenti.
    # Garantisco che post con lo stesso argomento aggancino lo stesso nodo Topic,
    # così la gap-analysis riconosce i doppioni.
    canon = canonical_topic(topic) or canonical_topic(state.get("research_brief", ""))
    print(f"\nSalvataggio del post '{post_title}' (topic canonico: '{canon}') nel Knowledge Graph.")
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
    print(f"\n{result}")

    # Il post e' pubblicato: lo 'promuovo' rimuovendolo dal backlog delle proposte
    # (se proveniva dal piano). Cosi' non ricompare tra le proposte pendenti.
    try:
        print(f"\n{remove_proposal(canon)}")
    except Exception as e:
        print(f"\nRimozione proposta non riuscita ({e}).")

    # Non termino: vado al gate 'prossimo post'. Se restano post selezionati, l'utente
    # decide se continuare, con quale, o fermarsi (i rimanenti restano come proposte).
    return Command(goto="next_post_node", update={
        "status": "completed",
        "reasoning_trace": trace(state, "\nPost pubblicato (copertina + SEO + salvataggio KG)."),
    })



# HITL nel ciclo di scrittura multi-post)
# Dopo ogni post (pubblicato in update_kg_node o scartato in review_node) si passa di
# qui. Se restano post selezionati, l'agente si ferma e chiede all'utente se continuare,
# con quale post, o fermarsi. Se si ferma, i rimanenti restano salvati come proposte nel
# KG (gia' inseriti alla selezione) ed e' tutto recuperabile. Niente avanzamento
# automatico: e' l'utente a guidare il ritmo (gestione dei tempi di esecuzione locale).


# Sceglie quale post della coda scrivere in base alla risposta dell'utente:
# un numero esplicito, un match sul topic, altrimenti il primo.
def _pick_next_index(text: str, queue: list) -> int:
    if not text or not queue:
        return 0
    low = text.lower()
    m = re.search(r"\b(\d+)\b", low)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(queue):
            return idx
    for i, p in enumerate(queue):
        topic = (p.get("topic", "") or "").lower()
        if topic and (topic in low or any(w in low for w in topic.split()[:2] if len(w) > 3)):
            return i
    return 0


def next_post_node(state: dict) -> Command[Literal["research_agent", "__end__"]]:
    queue = list(state.get("selected_posts") or [])
    if not queue:
        print("\nTutti i post selezionati sono stati gestiti. Chiudo.")
        return Command(goto=END, update={"status": "completed_all"})

    listing = "\n".join(
        f"{i + 1}. [{p.get('post_category') or p.get('category') or 'n/d'}] {p.get('topic', '')}"
        for i, p in enumerate(queue)
    )
    description = (
        "# Post selezionati ancora da scrivere\n\n" + listing +
        "\n\nVuoi continuare? Indica QUALE scrivere ora (es. \"scrivi il 2\", oppure "
        "\"continua\" per il primo), oppure fermati: i rimanenti restano salvati come proposte."
    )
    request = {
        "action_request": {"action": "continue_writing", "args": {}},
        "config": {
            "allow_accept": True, "allow_respond": True,
            "allow_ignore": True, "allow_edit": False,
        },
        "description": description,
    }
    answer = interrupt(request)
    rtype = answer.get("type") if isinstance(answer, dict) else None

    if rtype == "ignore":
        print("\nMi fermo qui: i post rimasti restano salvati come proposte nel KG.")
        return Command(goto=END, update={"status": "stopped_with_pending"})

    # accetto il primo post della coda e capisco quale post scrivere ora
    chosen = 0
    if rtype == "response":
        text = (answer.get("args", "") if isinstance(answer.get("args"), str)
                else str(answer.get("args", "")))
        chosen = _pick_next_index(text, queue)

    next_post = queue.pop(chosen)
    reset = _reset_for_new_post(state, next_post)
    reset["selected_posts"] = queue
    reset["reasoning_trace"] = trace(state, f"Proseguo col prossimo post: {next_post.get('topic', '')}.")
    print(f"\nProseguo con: {next_post.get('topic', '')}")
    return Command(goto="research_agent", update=reset)


# Questo nodo sostituisce il ToolNode di LangGraph standard. Quando il modello
# chiama un tool errato o solleva un'eccezione, questo nodo non blocca il grafo
# ma restituisce un ToolMessage di errore, in modo che l'agente possa correggersi nel
# turno successivo.

def resilient_tool_node(state: dict):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    outputs = []
    raw_notes_collected = []
    available = ", ".join(tools_by_name.keys())

    # Tool da cui recuperare le fonti (raw_notes) per il grounding. 
    # Sono esclusi i tool che non ritornano dati utili al grounding
    # come i tool di sola lettura del KG.
    _GROUNDING_TOOL_NAMES = {
        "mcp_web_search", "fetch_vehicle_specs", "compare_vehicles",
        "compare_vehicles_tool", "fetch_automotive_trends", "retrieve_local_documents",
    }

    # Limite di ricerche web per ciclo. Ogni ricerca impiega circa 40/60 secondi.
    # Se non limitassi questa cosa, l'agente potrebbe continuamente chiamare il tool
    # di ricerca web (perché autogiudica le fonti come non attendibili) andando in loop.
    # Una volta esaurite le ricerche, l'agente è costretto a lavorare con ciò che ha.
    # Il contatore lo azzero quando arriviamo nel nodo revisione e in particolare
    # quando l'utente vuole elementi aggiuntivi nel post (e non un nuovo drafting).
    MAX_WEB_SEARCHES = 2
    web_done = state.get("web_search_count")
    if web_done is None:
        web_done = sum(
            1 for m in state.get("messages", [])
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "mcp_web_search"
        )

    # Registro delle chiamate gia' eseguite (nome tool + argomenti normalizzati):
    # se il modello richiama un tool con gli stessi argomenti, blocco la ripetizione.
    # Se non ha funzionato la prima volta, non funzionera' nemmeno la seconda.
    done_calls = list(state.get("done_tool_calls") or [])

    for call in tool_calls:
        name = call.get("name")
        call_id = call.get("id")
        args = call.get("args", {}) or {}
        tool = tools_by_name.get(name)

        if config.debug:
            print(f"\nChiamato il tool: '{name}' | args={args}")


        args = normalize_tool_args(name, args)

        # Spesso il modelo omette il brand, significa che quando
        # chiama il fetch_vehicle_specs, non mette il brand nel car_model.
        # Per ovviare a questo problema, cerco il topic e provo a estrarre il brand.
        # Filtro le parole italiane comuni e tra i candidati rimasti scelgo quello
        # più vicino al modello nel testo. 
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
                    # Raccogliamo tutti i possibili candidati al brand.
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
                        # Scegliamo il candidato più vicino al codice modello nel topic
                        model_pos = topic.lower().find(car_model.lower())
                        best = min(candidates,
                                   key=lambda c: abs(topic.lower().find(c.lower()) - model_pos))
                        enriched = f"{best} {car_model}"
                        print(f"Aggiunto il brand al modello: '{car_model}' -> '{enriched}' (brand dal topic).")
                        args = {**args, "car_model": enriched}

        # Applica il tetto alle ricerche web
        if name == "mcp_web_search" and web_done >= MAX_WEB_SEARCHES:
            content = (
                f"Limite di {MAX_WEB_SEARCHES} ricerche web raggiunto per questo post. "
                "Usa le informazioni gia' raccolte (fonti locali e web) per scrivere la bozza, "
                "senza altre ricerche."
            )
            print(f"\nRicerca web saltata perché il limite di {MAX_WEB_SEARCHES} ricerche è stato raggiunto.")
            outputs.append(ToolMessage(content=content, name=name, tool_call_id=call_id))
            continue

        # Blocco delle chiamate ripetute: stessa coppia (tool, argomenti) gia' eseguita
        # in questo giro. Ripetere una chiamata identica non puo' dare un esito diverso:
        # rispondo subito istruendo il modello a usare quanto gia' raccolto o a cambiare
        # strategia, senza consumare tempo (fetch_vehicle_specs impiega anche 30s).
        try:
            signature = f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        except Exception:
            signature = f"{name}|{str(sorted(args.items()))}"
        if signature in done_calls:
            content = (
                f"Il tool '{name}' e' GIA' stato chiamato con questi stessi argomenti e "
                "l'esito lo hai gia' ricevuto. NON ripetere la stessa chiamata: usa le "
                "informazioni gia' raccolte, oppure cambia tool o argomenti, oppure procedi "
                "alla stesura con cio' che hai."
            )
            print(f"\nChiamata ripetuta a '{name}' con gli stessi argomenti: bloccata.")
            outputs.append(ToolMessage(content=content, name=name, tool_call_id=call_id))
            continue
        done_calls.append(signature)

        if tool is None:
            content = (
                f"ERRORE: il tool '{name}' non esiste. "
                f"Usa SOLO uno di questi tool, con il nome ESATTO: {available}."
            )
            print(f"\nNome tool inesistente '{name}'.")
        else:
            try:
                content = str(tool.invoke(args))
                if name == "mcp_web_search":
                    web_done += 1
            except Exception as e:
                content = f"Errore durante l'esecuzione del tool '{name}': {e}. Riprova o usa un altro tool."
                print(f"\nEccezione nel tool '{name}': {e}")

        # Anteponiamo l'etichetta "Observation (<tool>):" al risultato: cosi' l'osservazione
        # e' parte esplicita del flusso di messaggi (la vede il modello al passo successivo
        # ed e' visibile in LangSmith come messaggio tool), chiudendo il ciclo ReAct in modo
        # leggibile. NB: i messaggi di servizio (limite/blocco) sono creati altrove e NON
        # passano di qui, quindi i controlli startswith del grader restano validi.
        msg = ToolMessage(content=f"Observation ({name or 'tool'}): {content}",
                          name=name or "unknown", tool_call_id=call_id)
        outputs.append(msg)

        # Se il tool è una fonte (grounding) allora salvo l'output grezzo del tool
        # dentro le raw_notes per poi usarle nel nodo di stesura post.
        if name in _GROUNDING_TOOL_NAMES and content.strip() and "errore" not in content.lower()[:40]:
            raw_notes_collected.append(f"[{name}] {content}")

    new_state = {"messages": outputs, "web_search_count": web_done, "done_tool_calls": done_calls}
    if raw_notes_collected:
        existing = state.get("raw_notes") or []
        new_state["raw_notes"] = existing + raw_notes_collected

    # Registro le OBSERVATION nel reasoning trace: chiude il ciclo ReAct esplicito
    # (Thought -> Action -> Observation) richiesto dalle specifiche. Il Thought e
    # l'Action vengono gia' tracciati in research_agent_node; qui aggiungo l'esito
    # sintetico di ogni tool eseguito, cosi' la giustificazione e l'effetto di ogni
    # invocazione sono leggibili nel trace senza dover scorrere i messages.
    if outputs:
        obs_lines = []
        for out_msg in outputs:
            name = getattr(out_msg, "name", "tool")
            content = str(getattr(out_msg, "content", ""))[:220].replace("\n", " ")
            obs_lines.append(f"Observation ({name}): {content}")
        new_state["reasoning_trace"] = trace(state, "\n".join(obs_lines))
    return new_state
