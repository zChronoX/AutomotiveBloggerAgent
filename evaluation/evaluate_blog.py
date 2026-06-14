"""
File di valutazione e osservabilità dell'agente per essere usata su LangSmith.
Copre le specifiche richieste per l'evaluation:
  1. qualitative analysis of generated posts : evaluate_quality
  2. assessment of source quality and grounding : evaluate_grounding
  3. identification of at least three failure cases: evaluate_failure_cases
La valutazione usa l'osservabilita' di LangSmith.
Il grafo viene fermato all'interrupt della review, quindi non viene mai modificato,
da eventuali approvazioni o modifiche HITL, quindi la valutazione può essere eseguita
più volte senza resettare il KG.
L'orchestrazione avviene tramite 'evaluate' di LangSmith, in cui l'output che sarebbe la bozza
o le proposte editoriali, vengono passate agli evaluator che valutano la qualità, grounding,
casi di fallimento e uso dei tool. Il tutto visualizzabile durante e dopo l'esecuzione
nella dashboard di LangSmith.
"""

import os
import sys


# Import di moduli che si trovano nella root del progetto, quindi aggiungo la cartella al path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from langsmith import Client
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
client = Client()

# Per la valutazione ho deciso di usare un'approccio misto. 
# Usare un giudice locale oppure lo stesso modello, non era l'idea migliore
# per cui ho optato per usare Gemini 3.1 Flash Lite come giudice principale
# il problema è che Gemini ha un limite di richieste che posso fare giornalmente
# pertanto, nel caso in cui dovesse dare errore, ho inserito un fallback ad un giudice
# locale che è sempre Ministral 3.
def _build_evaluator_llm():
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite",
                temperature=0.0,
                google_api_key=api_key,
                max_retries=3,          # Gestisco il "too many request (429)"
            )
            print("Uso Gemini 3.1 Flash Lite come valutatore.")
            return llm
        except Exception as e:
            print(f"Gemini non inizializzabile ({e}). Fallback a Ministral locale.")
    else:
        print("GOOGLE_API_KEY assente: uso Ministral 3B locale come valutatore.")
    return ChatOllama(model="ministral-3:3b", temperature=0.0)


evaluator_llm = _build_evaluator_llm()
_is_primary_local = isinstance(evaluator_llm, ChatOllama)
_fallback_llm = None if _is_primary_local else ChatOllama(model="ministral-3:3b", temperature=0.0)


def _graded_invoke(schema, prompt_text):
    """
    Chiamo Gemini come giudice principale, se non dovesse funzionare,
    per problemi di limiti, API KEY o altro, uso Ministral.
    Restituisce l'oggetto strutturato, o None se entrambi falliscono.
    """
    try:
        return evaluator_llm.with_structured_output(schema).invoke(prompt_text)
    except Exception as e_primary:
        if _fallback_llm is not None:
            try:
                print(f"Errore con ({e_primary}); uso Ministral di riserva.")
                return _fallback_llm.with_structured_output(schema).invoke(prompt_text)
            except Exception:
                return None
        return None

# Dataset da valutare. Nel mio caso ho due dataset, il primo più piccolo
# per testare il funzionamento globale dell'agente (con tutti i tool)
# il secondo per raccogliere dati e metriche più precise per
# la parte di evaluation.
DATASET_NAME = os.environ.get("EVAL_DATASET", "AutomotiveBloggerAgent V1.1")


# Legge l'interrupt attualmente pendente dallo snapshot del grafo.
# Gli interrupt LangGraph espongono il payload passato a interrupt(...) in
# task.interrupts[].value: da li' ricaviamo l'"action" per decidere come riprendere.
def _pending_interrupt_action(snapshot):
    for task in (getattr(snapshot, "tasks", None) or ()):
        for itr in (getattr(task, "interrupts", None) or ()):
            val = getattr(itr, "value", None)
            if isinstance(val, dict):
                return val.get("action_request", {}).get("action")
    return None


# Funzione principale di esecuzione della valutazione, si ferma all'interrupt della bozza.
def run_blogger_until_draft(inputs: dict) -> dict:
    """
    Funzione-target per LangSmith evaluate().
    Esegue il grafo fino all'interrupt della review della BOZZA e restituisce la bozza,
    SENZA approvarla (quindi senza aggiornare il Knowledge Graph).

    NB: dopo l'introduzione del planning multi-post, il PRIMO interrupt del grafo non e'
    piu' la review della bozza ma il GATE EDITORIALE (review_editorial_plan), che scatta
    prima della ricerca e della stesura. Per i prompt di scrittura quindi non basta un
    singolo invoke: bisogna attraversare i gate di planning (e l'eventuale richiesta di
    chiarimento) riprendendo il grafo, fino a raggiungere la bozza. Qui lo facciamo in
    automatico, accettando tutte le proposte del piano ("scrivili tutti") e senza scrivere
    post aggiuntivi. I prompt del ramo suggerimenti non passano dai gate e terminano subito.
    """
    import uuid
    from agent import graph
    from langgraph.types import Command

    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    user_input = inputs.get("user_input", "")

    # Numero massimo di riprese, per non rischiare loop: chiarimento + gate editoriale +
    # margine. Arrivati alla bozza ci fermiamo, quindi in pratica ne bastano 1-2.
    MAX_RESUMES = 6
    snapshot = None
    error = None

    try:
        # Avvio del grafo: si fermera' al primo interrupt (gate editoriale per i prompt di
        # scrittura, chiarimento se la richiesta e' vaga) oppure terminera' (suggerimenti).
        graph.invoke({"user_input": user_input}, config)

        for _ in range(MAX_RESUMES):
            snapshot = graph.get_state(config)
            # Grafo terminato (nessun nodo pendente): es. ramo suggerimenti o annullamento.
            if not (snapshot and snapshot.next):
                break

            action = _pending_interrupt_action(snapshot)

            # Siamo all'interrupt della BOZZA: e' esattamente dove vogliamo fermarci.
            if action == "review_post_draft":
                break

            # Altrimenti attraversiamo i gate di planning senza modificare il KG:
            if action == "review_editorial_plan":
                # accetto TUTTE le proposte del piano -> il grafo prosegue verso la stesura
                resume = {"type": "response", "args": "scrivili tutti"}
            elif action == "clarify_request":
                # non blocco la valutazione su un chiarimento: procedo con la richiesta originale
                resume = {"type": "ignore"}
            elif action == "continue_writing":
                # in valutazione non scrivo post successivi: mi fermo qui
                resume = {"type": "ignore"}
            elif action == "choose_suggestion":
                # gate dei suggerimenti: in valutazione non scelgo nulla da scrivere,
                # cosi' la run termina con il testo dei suggerimenti come output.
                resume = {"type": "ignore"}
            else:
                # interrupt non riconosciuto: mi fermo per non rischiare un loop
                break

            graph.invoke(Command(resume=resume), config)
    except Exception as e:
        error = f"Esecuzione grafo fallita: {e}"

    # Leggiamo lo stato corrente (anche in caso di errore, per restituire cio' che c'e').
    try:
        snapshot = graph.get_state(config)
    except Exception:
        snapshot = None
    values = snapshot.values if snapshot else {}

    draft = values.get("draft_content") or ""
    # Se non c'e' una bozza, come nel caso della richiesta di suggerimenti, usiamo l'ultimo messaggio
    if not draft:
        msgs = values.get("messages", [])
        if msgs:
            draft = getattr(msgs[-1], "content", "") or ""

    # Raccogliamo i nomi dei tool effettivamente usati durante la run.
    # Li prendiamo dagli AIMessage all'interno di "tool_calls" 
    tools_called = []
    for m in values.get("messages", []):
        for tc in (getattr(m, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tools_called.append(name)

    # Questi tool non compaiono nelle tool calls, perché sarebbero tool deterministici (parliamo del RAG e del KG)
    # ma li tiriamo fuori dallo stato e da li vediamo se sono stati chiamati o no.
    if values.get("kg_summary"):
        tools_called.append("query_knowledge_graph")
    if values.get("local_sources"):
        tools_called.append("retrieve_local_documents")
    if values.get("trends_summary"):
        tools_called.append("fetch_automotive_trends")

    return {
        "draft_content": draft,
        "status": values.get("status", ""),
        "sources": values.get("sources", []) or [],
        "local_sources": values.get("local_sources", []) or [],
        "interrupted": bool(snapshot.next) if snapshot else False,
        "tools_called": tools_called,
        "user_input": user_input,
        "error": error,
    }


def _extract_post(run) -> str:
    """Estrae il testo del post dall'output della target_fn."""
    try:
        out = run.outputs or {}
        return out.get("draft_content") or ""
    except Exception:
        return str(getattr(run, "outputs", ""))


# Classe per valutare la qualità del post
class QualityScore(BaseModel):
    score: int = Field(description="Punteggio da 1 a 5. 5 significa eccellente.")
    reasoning: str = Field(description="Motivazione del punteggio.")

# Primo evaluator: la quality, misura la struttura, leggibilità, tono, ecc. del post
# Chiedo al giudice di dare un voto da 1 a 5 basandomi sulla struttura del Markdown, assenza di ripetizioni, tono ingaggiante e professionalita'.
def evaluate_quality(run, example) -> dict:
    generated_post = _extract_post(run)
    if not generated_post:
        return {"key": "Qualitative_Score", "score": 0.0, "comment": "Nessuna bozza generata."}

    grader_prompt = (
        "Sei un editor giornalistico. Valuta qualitativamente il seguente post da 1 a 5.\n"
        "Criteri: struttura Markdown pulita, assenza di ripetizioni, tono ingaggiante e professionalita'.\n"
        f"Articolo:\n{generated_post}\n"
    )
    result = _graded_invoke(QualityScore, grader_prompt)
    if result is None:
        return {"key": "Qualitative_Score", "score": 0.0, "comment": "Giudice non disponibile (principale e fallback falliti)."}
    return {"key": "Qualitative_Score", "score": result.score / 5.0, "comment": result.reasoning}


# Classe per la valutazione del grounding.
class GroundingScore(BaseModel):
    factualita: int = Field(description="Da 0 a 5: quanto il testo e' fattuale e privo di speculazioni inventate (5 = pienamente fattuale).")
    reasoning: str = Field(description="Spiega se le fonti sono usate correttamente o se ci sono affermazioni non supportate.")




# Secondo evaluator: il grounding. Valuta le fonti, cercando di capire se sono fonti reali o inventate.
# Cerco marcatori come [FONTE], URL, "http", ecc. Se li trovo do 1.0 altrimenti 0.0
# Secondariamente verifico la fattualità, quindi controllo se nel testo ci sono parole come
#"si prevede", "si prevede che", "potrebbe", "probabilmente". Se ci sono, do meno punti.
# lo score finale è un 50% di entrambi.


def evaluate_grounding(run, example) -> dict:
    generated_post = _extract_post(run)
    if not generated_post:
        return {"key": "Source_Grounding", "score": 0.0, "comment": "Nessuna bozza generata."}

    # (a) Check DETERMINISTICO delle citazioni: oggettivo, non soggetto al giudizio del 3B.
    low = generated_post.lower()
    citation_markers = ["http", "[fonte", "fonte:", "fonti:", "secondo", "[1]", "riferimenti", ".txt", ".it", ".com"]
    has_citations = any(m in low for m in citation_markers)
    cit_score = 1.0 if has_citations else 0.0

    # (b) Giudizio del modello sulla FATTUALITA' (0-5), separato dalle citazioni.
    grader_prompt = (
        "Valuta da 0 a 5 quanto il seguente articolo automotive e' FATTUALE, cioe' privo di "
        "affermazioni palesemente inventate o speculazioni prive di base. NON penalizzare le "
        "normali espressioni di previsione ('si prevede', 'potrebbe') tipiche del giornalismo: "
        "penalizza solo dati o fatti chiaramente falsi o non supportati.\n"
        "5 = pienamente fattuale; 0 = pieno di invenzioni.\n\n"
        f"Articolo:\n{generated_post[:6000]}\n"
    )
    result = _graded_invoke(GroundingScore, grader_prompt)
    if result is not None:
        fact_score = max(0, min(5, result.factualita)) / 5.0
        reasoning = result.reasoning
    else:
        fact_score = 0.5  # se entrambi i giudici falliscono, valore neutro invece di penalizzare
        reasoning = "(giudizio fattualita' non disponibile)"

    final = round(0.5 * cit_score + 0.5 * fact_score, 2)
    comment = (f"Citazioni presenti: {'si' if has_citations else 'NO'} "
               f"(peso 0.5). Fattualita': {fact_score:.1f} (peso 0.5). {reasoning}")
    return {"key": "Source_Grounding", "score": final, "comment": comment}


# Terzo evaluator: casi di fallimento, ne misuro 4, bozza vuota, non ci sono citazioni, bozza esistente ma non ci sono tool
# quindi grounding assente (il modello ha inventato tutto), e poi il caso di fallimento completo dell'agente
# non c'è una bozza e non si è fermato all'interrupt, si è bloccato al grafo (può succedere).
# Se tutti passano do 1.0, se uno di questi fallisce, scalo il punteggio di 0,25.
def evaluate_failure_cases(run, example) -> dict:

    out = run.outputs or {}
    post = out.get("draft_content") or ""
    sources = out.get("sources") or []
    interrupted = out.get("interrupted", False)

    failures = []
    # F1
    if not post.strip():
        failures.append("F1_EMPTY_DRAFT: nessuna bozza generata.")
    # F2 (solo se c'e' una bozza da controllare)
    citation_markers = ["[fonte", "fonte:", "http", "secondo la fonte", "[1]", "riferimenti:", "fonti:"]
    if post.strip() and not any(mark in post.lower() for mark in citation_markers):
        failures.append("F2_NO_CITATIONS: bozza priva di citazioni esplicite.")
    # F3
    if post.strip() and not sources:
        failures.append("F3_NO_TOOL_USAGE: nessuna fonte raccolta dai tool di ricerca.")
    # F4
    if not post.strip() and not interrupted:
        failures.append("F4_HUNG_NO_INTERRUPT: nessun output e nessuna interruzione di review.")

    total_checks = 4
    passed = total_checks - len(failures)
    comment = "Nessun fallimento rilevato." if not failures else " | ".join(failures)
    return {"key": "Failure_Cases", "score": passed / total_checks, "comment": comment}


# Quarto evaluator: uso dei tool. Controllo se ha usato almeno un tool, successivamente
# se l'ha usato, vedo se è un tool di ricerca, e poi il caso specifico del "compare_vehicles",
# in cui controllo se è stato richiesto un confronto specifico, ma quel tool non è stato chiamato.
def evaluate_tool_usage(run, example) -> dict:
    out = run.outputs or {}
    tools_called = out.get("tools_called") or []
    user_input = (out.get("user_input") or "").lower()

    # Categorie di tool "di ricerca/grounding": controllo se la ricerca web o kg sono stati chiamati
    research_tools = {"retrieve_local_documents", "mcp_web_search",
                      "fetch_automotive_trends", "query_knowledge_graph",
                      "get_editorial_context", "fetch_vehicle_specs"}
    used_research = [t for t in tools_called if t in research_tools]

    # Controllo che il tool di comparazione è stato usato in base alla richiesta
    # se non è stato usato, lo aggiungo alle note ma non penalizzo il tool usage
    # perché comunque le informazioni le avrà tirate fuori da una ricerca web o dal tool delle specifiche
    # quindi senza allucinazioni.
    is_comparison = any(k in user_input for k in ("confront", "paragon", "differenz", " vs ", "meglio"))

    notes = []
    score = 1.0

    if not tools_called:
        score = 0.3
        notes.append("Nessun tool invocato dall'agente.")
    else:
        notes.append(f"Tool usati: {', '.join(tools_called)}.")
        if not used_research:
            score = 0.6
            notes.append("Nessun tool di ricerca/grounding tra quelli usati.")
        if is_comparison and "compare_vehicles" not in tools_called:
            notes.append("Richiesta di confronto ma 'compare_vehicles' non usato.")

    return {"key": "Tool_Usage", "score": score, "comment": " ".join(notes)}


# Report di osservabilità: misuro le metriche estratte da LangSmith
# come errori, step medi del grafo e latenza.
#CAMBIARE IL NOME DEL PROGETTO
def observability_report(project_name: str = None, limit: int = 50) -> dict:
    project = project_name or os.environ.get("LANGSMITH_PROJECT", "blogger-copilot")
    try:
        # L'API di LangSmith accetta al massimo 100 run per richiesta, sennò da errore.
        fetch_limit = min(limit * 20, 100)
        all_runs = list(client.list_runs(project_name=project, limit=fetch_limit))
    except Exception as e:
        return {"error": f"Impossibile leggere le run da LangSmith: {e}"}

    if not all_runs:
        return {"info": f"Nessuna run trovata nel progetto '{project}'."}

    # Uso le run ROOT sono le esecuzioni complete del grafo.
    root_runs = [r for r in all_runs if getattr(r, "is_root", False)]
    # Se l'attributo is_root non e' disponibile, usiamo quelle senza parent.
    if not root_runs:
        root_runs = [r for r in all_runs if not getattr(r, "parent_run_id", None)]

    latencies, errors = [], 0
    for r in root_runs:
        if getattr(r, "start_time", None) and getattr(r, "end_time", None):
            latencies.append((r.end_time - r.start_time).total_seconds())
        if getattr(r, "error", None):
            errors += 1

    n_root = len(root_runs) or 1
    n_all = len(all_runs)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_steps = round(n_all / n_root, 2)
    return {
        "project": project,
        "esecuzioni_grafo (root)": n_root,
        "run_totali (nodi+step)": n_all,
        "latenza_media_s": round(avg_latency, 2),
        "run_con_errori": errors,
        "tasso_errore": round(errors / n_root, 3),
        "step_medi_per_esecuzione": avg_steps,
    }


# Verifico che LangSmith sia configurato e poi chiamo la funzione evaluate di LangSmith.
if __name__ == "__main__":
    from config import check_langsmith_setup
    from langsmith.evaluation import evaluate

    # Posso impostare il dataset da riga di comando, sennò prendo quello di default sopra.
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DATASET_NAME

    print("Evaluation AutomotiveBloggerAgent")
    if not check_langsmith_setup():
        print("LangSmith non configurato: la valutazione puo' girare ma senza tracing/dashboard.")

    print(f"\nAvvio valutazione sul dataset '{dataset_name}'")
    try:
        results = evaluate(
            run_blogger_until_draft,
            data=dataset_name,
            evaluators=[evaluate_quality, evaluate_grounding, evaluate_failure_cases, evaluate_tool_usage],
            experiment_prefix="AutomotiveBloggerAgent",
        )
        print("Valutazione completata e risultati sulla dashboard LangSmith.")
    except Exception as e:
        print(f"Errore durante evaluate(): {e}")
        print("Manca il dataset o Ollama non avviato.")

    print("\nParametri di osservabilità: \n")
    report = observability_report()
    for k, v in report.items():
        print(f"  {k}: {v}")
