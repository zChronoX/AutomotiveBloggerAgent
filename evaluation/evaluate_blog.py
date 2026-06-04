"""
Suite di valutazione del Blogger Copilot su LangSmith.

Copre i tre assi richiesti dalle specifiche (Evaluation Requirement):
  1. qualitative analysis of generated posts      -> evaluate_quality
  2. assessment of source quality and grounding   -> evaluate_grounding
  3. identification of at least three failure cases-> evaluate_failure_cases
La valutazione si appoggia all'osservabilita' di LangSmith: ogni run del grafo e'
tracciata (vedi .env: LANGSMITH_TRACING/API_KEY/PROJECT) e da quelle run estraiamo i
parametri di osservabilita' (latenza, numero di step, errori) con observability_report().

Pattern (evaluator che ricevono run/example e tornano {key, score, comment} + evaluate())
ripreso dai test del tutorial "agents-from-scratch".

NOTA HITL: in valutazione il grafo viene fermato all'interrupt della review e si legge
'draft_content' dallo stato. NON si riprende il grafo, quindi il Knowledge Graph (Neo4j)
NON viene mai modificato dai test: la valutazione e' ripetibile e senza effetti collaterali.

Uso (dalla radice del progetto):
    python -m evaluation.evaluate_blog
"""

import os
import sys

# Questo script vive nella sottocartella evaluation/, ma importa moduli che stanno nella
# RADICE del progetto (config, agent, ecc.). Aggiungiamo la cartella genitore al path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from langsmith import Client
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
client = Client()

# Giudice deterministico (temperature=0.0): la valutazione deve essere riproducibile.
evaluator_llm = ChatOllama(model="ministral-3:3b", temperature=0.0)

# Dataset da valutare. Si puo' scegliere da riga di comando o da .env (EVAL_DATASET):
#   python -m evaluation.evaluate_blog                  -> usa il default qui sotto
#   python -m evaluation.evaluate_blog "Nome Dataset"   -> usa quello passato
# Cosi' puoi lanciare prima il V1.1 (5 prompt) e poi il V2 (15 prompt) senza modificare il codice.
DATASET_NAME = os.environ.get("EVAL_DATASET", "Blogger_Test_Dataset V1.1")


# ============================================================
# TARGET FUNCTION: invoca il grafo e si ferma alla bozza (HITL)
# ============================================================
def run_blogger_until_draft(inputs: dict) -> dict:
    """
    Funzione-target per LangSmith evaluate().
    Esegue il grafo fino all'interrupt della review umana e restituisce la bozza,
    SENZA riprendere (quindi senza aggiornare il Knowledge Graph).
    """
    import uuid
    from agent import graph

    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    user_input = inputs.get("user_input", "")

    # Avvia il grafo: si fermera' all'interrupt() dentro review_node (se arriva a quel punto).
    try:
        graph.invoke({"user_input": user_input}, config)
    except Exception as e:
        return {"draft_content": "", "error": f"Esecuzione grafo fallita: {e}"}

    # Leggiamo lo stato corrente (bozza + eventuali campi utili agli evaluator)
    snapshot = graph.get_state(config)
    values = snapshot.values if snapshot else {}

    draft = values.get("draft_content") or ""
    # Fallback: se non c'e' una bozza (es. richiesta di soli "topic"), usiamo l'ultimo messaggio
    if not draft:
        msgs = values.get("messages", [])
        if msgs:
            draft = getattr(msgs[-1], "content", "") or ""

    # Raccogliamo i NOMI DEI TOOL effettivamente usati durante la run.
    # (a) Tool SCELTI dal modello (ReAct): compaiono come tool_calls negli AIMessage.
    tools_called = []
    for m in values.get("messages", []):
        for tc in (getattr(m, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tools_called.append(name)

    # (b) Tool DETERMINISTICI (K-RAG): KG e RAG locale sono invocati dal codice, NON dal
    # modello, quindi non compaiono nei tool_calls. Li deduciamo dallo stato: se c'e' contesto
    # KG o documenti locali, quei tool sono stati eseguiti. Senza questo, l'evaluator tool-usage
    # sottostimerebbe (segnalando "nessun tool" anche con K-RAG attivo).
    if values.get("kg_summary"):
        tools_called.append("query_knowledge_graph")
    if values.get("local_sources"):
        tools_called.append("retrieve_local_documents")

    return {
        "draft_content": draft,
        "status": values.get("status", ""),
        "sources": values.get("sources", []) or [],
        "local_sources": values.get("local_sources", []) or [],
        "interrupted": bool(snapshot.next) if snapshot else False,
        "tools_called": tools_called,
        "user_input": user_input,
    }


def _extract_post(run) -> str:
    """Estrae il testo del post dall'output della target_fn (con fallback robusti)."""
    try:
        out = run.outputs or {}
        return out.get("draft_content") or ""
    except Exception:
        return str(getattr(run, "outputs", ""))


# ============================================================
# 1. QUALITATIVE ANALYSIS
# ============================================================
class QualityScore(BaseModel):
    score: int = Field(description="Punteggio da 1 a 5. 5 significa eccellente.")
    reasoning: str = Field(description="Motivazione del punteggio.")


def evaluate_quality(run, example) -> dict:
    """Valuta struttura, leggibilita' e interesse del post generato."""
    generated_post = _extract_post(run)
    if not generated_post:
        return {"key": "Qualitative_Score", "score": 0.0, "comment": "Nessuna bozza generata."}

    prompt = PromptTemplate.from_template(
        "Sei un editor giornalistico. Valuta qualitativamente il seguente post da 1 a 5.\n"
        "Criteri: struttura Markdown pulita, assenza di ripetizioni, tono ingaggiante e professionalita'.\n"
        "Articolo:\n{post}\n"
    )
    grader = evaluator_llm.with_structured_output(QualityScore)
    try:
        result = grader.invoke(prompt.format(post=generated_post))
        return {"key": "Qualitative_Score", "score": result.score / 5.0, "comment": result.reasoning}
    except Exception as e:
        return {"key": "Qualitative_Score", "score": 0.0, "comment": f"Errore valutazione: {e}"}


# ============================================================
# 2. SOURCE GROUNDING & CITATIONS (K-RAG)
# ============================================================
class GroundingScore(BaseModel):
    score: int = Field(description="1 se i fatti sono supportati e ci sono citazioni esplicite, 0 altrimenti.")
    reasoning: str = Field(description="Spiega se le fonti sono citate correttamente o se ci sono allucinazioni.")


def evaluate_grounding(run, example) -> dict:
    """Valuta ancoraggio alle fonti e presenza di citazioni esplicite."""
    generated_post = _extract_post(run)
    if not generated_post:
        return {"key": "Source_Grounding", "score": 0, "comment": "Nessuna bozza generata."}

    prompt = PromptTemplate.from_template(
        "Analizza il seguente articolo. Requisiti fondamentali:\n"
        "1. Ci sono citazioni o riferimenti espliciti alle fonti (es. [1], 'Secondo la fonte...', link)?\n"
        "2. Il testo e' fattuale o contiene speculazioni/allucinazioni senza base tecnica?\n"
        "Assegna 1 solo se ENTRAMBE le condizioni sono soddisfatte, altrimenti 0.\n"
        "Articolo:\n{post}\n"
    )
    grader = evaluator_llm.with_structured_output(GroundingScore)
    try:
        result = grader.invoke(prompt.format(post=generated_post))
        return {"key": "Source_Grounding", "score": result.score, "comment": result.reasoning}
    except Exception as e:
        return {"key": "Source_Grounding", "score": 0, "comment": f"Errore: {e}"}


# ============================================================
# 3. FAILURE CASES (>= 3 casi di fallimento identificabili)
# ============================================================
def evaluate_failure_cases(run, example) -> dict:
    """
    Identifica in modo programmatico i casi di fallimento del sistema.
    Tre (o piu') failure case classificati:
      F1 - EMPTY_DRAFT      : il sistema non ha prodotto alcuna bozza.
      F2 - NO_CITATIONS     : la bozza non contiene citazioni/riferimenti (grounding mancante).
      F3 - NO_TOOL_USAGE    : nessuna fonte raccolta (l'agente non ha usato i tool di ricerca).
      F4 - HUNG_NO_INTERRUPT: il grafo non si e' fermato alla review ne' ha prodotto output utile.
    Score = frazione di controlli SUPERATI (1.0 = nessun fallimento).
    """
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


# ============================================================
# 4. TOOL USAGE (tool-call test, in forma TOLLERANTE)
# ============================================================
def evaluate_tool_usage(run, example) -> dict:
    """Verifica, in modo tollerante, che l'agente abbia usato tool sensati per la richiesta."""
    out = run.outputs or {}
    tools_called = out.get("tools_called") or []
    user_input = (out.get("user_input") or "").lower()

    # Categorie di tool "di ricerca/grounding": per una richiesta di post ci aspettiamo
    # che ne sia stato usato ALMENO UNO (RAG locale OPPURE web OPPURE trends OPPURE KG).
    research_tools = {"retrieve_local_documents", "mcp_web_search",
                      "fetch_automotive_trends", "query_knowledge_graph",
                      "get_editorial_context", "fetch_vehicle_specs"}
    used_research = [t for t in tools_called if t in research_tools]

    # Comportamento atteso per categoria (SOFT): se la richiesta e' un confronto,
    # l'atteso ideale e' compare_vehicles; lo segnaliamo se assente, senza penalizzare a zero.
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
            notes.append("DIVERGENZA: nessun tool di ricerca/grounding tra quelli usati.")
        if is_comparison and "compare_vehicles" not in tools_called:
            notes.append("DIVERGENZA: richiesta di confronto ma 'compare_vehicles' non usato "
                         "(comportamento noto dei modelli locali piccoli).")

    return {"key": "Tool_Usage", "score": score, "comment": " ".join(notes)}


# ============================================================
# PARAMETRI DI OSSERVABILITA' (da LangSmith)
# ============================================================
def observability_report(project_name: str = None, limit: int = 50) -> dict:
    """
    Estrae i parametri di osservabilita' delle run dal progetto LangSmith:
    latenza media, errori, e numero medio di step per esecuzione del grafo.
    Soddisfa "in terms of observability parameters available in Langsmith".

    NOTA sul conteggio degli step: 'child_run_ids' spesso NON e' popolato da list_runs
    (per questo prima gli step risultavano 0). Contiamo invece in modo affidabile:
    - le run ROOT (is_root=True) = una per esecuzione del grafo;
    - TUTTE le run del progetto;
    e ricaviamo gli step medi come (run totali / run root): ogni nodo/sotto-step e' una run.
    """
    project = project_name or os.environ.get("LANGSMITH_PROJECT", "blogger-copilot")
    try:
        # L'API di LangSmith accetta al massimo 100 run per richiesta: non superare quel tetto.
        fetch_limit = min(limit * 20, 100)
        all_runs = list(client.list_runs(project_name=project, limit=fetch_limit))
    except Exception as e:
        return {"error": f"Impossibile leggere le run da LangSmith: {e}"}

    if not all_runs:
        return {"info": f"Nessuna run trovata nel progetto '{project}'."}

    # Le run ROOT sono le esecuzioni complete del grafo (quelle che ci interessano per
    # latenza ed errori a livello di richiesta utente).
    root_runs = [r for r in all_runs if getattr(r, "is_root", False)]
    # Fallback: se l'attributo is_root non e' disponibile, usiamo quelle senza parent.
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
    # Step medi per esecuzione: tutte le run (nodi + sotto-step) diviso le esecuzioni complete.
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


# ============================================================
# ESECUZIONE
# ============================================================
if __name__ == "__main__":
    from config import check_langsmith_setup
    from langsmith.evaluation import evaluate

    # Dataset: argomento da riga di comando se fornito, altrimenti il default (V1.1).
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DATASET_NAME

    print("== Suite di valutazione Blogger Copilot ==")
    if not check_langsmith_setup():
        print("LangSmith non configurato: la valutazione puo' girare ma senza tracing/dashboard.")

    print(f"\nAvvio valutazione sul dataset '{dataset_name}'...")
    try:
        results = evaluate(
            run_blogger_until_draft,
            data=dataset_name,
            evaluators=[evaluate_quality, evaluate_grounding, evaluate_failure_cases, evaluate_tool_usage],
            experiment_prefix="Blogger_Eval_Run",
        )
        print("Valutazione completata. Risultati sulla dashboard LangSmith.")
    except Exception as e:
        print(f"Errore durante evaluate(): {e}")
        print("Verifica che il dataset esista (esegui crea_dataset.py) e che Ollama/Neo4j siano attivi.")

    print("\n== Parametri di osservabilita' (LangSmith) ==")
    report = observability_report()
    for k, v in report.items():
        print(f"  {k}: {v}")
