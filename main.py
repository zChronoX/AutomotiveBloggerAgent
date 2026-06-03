"""Entry point del Blogger Copilot."""

import uuid
from langgraph.types import Command
from agent import graph
from config import check_langsmith_setup


def _print_header():
    print("\n" + "=" * 56)
    print("  BLOGGER COPILOT - Assistente editoriale automotive")
    print("=" * 56)
    print("Esempi di richiesta:")
    print("  - 'Scrivi un post sui SUV ibridi 2025'")
    print("  - 'Suggeriscimi argomenti per i prossimi post'")
    print("Scrivi 'esci' per chiudere.\n")


def _extract_interrupt(result):
    """
    Estrae il payload dell'interrupt dal risultato di graph.invoke().
    Restituisce il dict 'request' completo (con 'action_request' e 'description'),
    cosi' il chiamante puo' distinguere i due tipi di interrupt:
      - action 'clarify_request'   -> domanda di chiarimento (FASE 0 scoping)
      - action 'review_post_draft' -> revisione della bozza (FASE 4 HITL)
    Restituisce None se non c'e' interrupt.
    """
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    try:
        payload = interrupts[0].value
        request = payload[0] if isinstance(payload, list) else payload
        return request if isinstance(request, dict) else None
    except Exception:
        return None


def _interrupt_action(request) -> str:
    """Estrae il nome dell'azione da un request di interrupt."""
    try:
        return request.get("action_request", {}).get("action", "")
    except Exception:
        return ""



def _ask_review_choice() -> dict:
    """Menu numerato per la revisione umana. Ritorna il dict di resume per il grafo."""
    print("\n" + "-" * 56)
    print("[REVISIONE] Come vuoi procedere?")
    print("  1) Approva  -> salva il post nel Knowledge Graph")
    print("  2) Modifica -> richiedi cambiamenti e rigenera la bozza")
    print("  3) Scarta   -> annulla senza salvare")
    while True:
        scelta = input("Scelta [1/2/3]: ").strip()
        if scelta == "1":
            return {"type": "accept"}
        if scelta == "2":
            modifiche = input("Descrivi le modifiche da apportare: ").strip()
            return {"type": "response", "args": modifiche}
        if scelta == "3":
            return {"type": "ignore"}
        print("Scelta non valida. Inserisci 1, 2 o 3.")


def _handle_clarification(request, config):
    """
    Gestisce l'interrupt di chiarimento (FASE 0 scoping): mostra la domanda
    dell'agente, raccoglie la risposta dell'utente e riprende il grafo.
    Ritorna il nuovo 'result' dopo il resume.
    """
    question = request.get("description", "Puoi chiarire meglio la tua richiesta?")
    print("\n" + "-" * 56)
    print("[CHIARIMENTO] L'agente ha bisogno di una precisazione:")
    print(f"  {question}")
    print("(Scrivi la tua risposta, oppure premi INVIO per procedere comunque.)")
    risposta = input("La tua risposta: ").strip()

    if not risposta:
        resume = {"type": "ignore"}  # procedi senza chiarimento
    else:
        resume = {"type": "response", "args": risposta}

    return graph.invoke(Command(resume=resume), config)


def _run_review_loop(config):
    """
    Gestisce il ciclo di revisione: finche' il grafo si ferma sulla review,
    mostra la bozza, chiede la scelta e riprende con Command(resume=...).
    Termina quando l'utente approva (salvataggio KG) o scarta.
    """
    while True:
        state = graph.get_state(config)
        if not state.next:
            # Il grafo e' arrivato alla fine (approvazione completata o flusso concluso)
            break

        scelta = _ask_review_choice()

        if scelta["type"] == "accept":
            print("\n[INFO] Approvazione ricevuta: aggiornamento del Knowledge Graph...")
        elif scelta["type"] == "ignore":
            print("\n[INFO] Bozza scartata: nessun salvataggio.")
        else:
            print("\n[INFO] Applico le modifiche e rigenero la bozza...")

        # Riprende il grafo iniettando la risposta dell'utente nell'interrupt
        result = graph.invoke(Command(resume=scelta), config)

        # Se dopo la ripresa il grafo si e' fermato di nuovo (caso 'modifica'),
        # mostriamo la nuova bozza e il loop continua.
        request = _extract_interrupt(result)
        if request and _interrupt_action(request) == "review_post_draft":
            print("\n" + "=" * 56)
            print("  NUOVA BOZZA (rivista) - pronta per la revisione")
            print("=" * 56)
            print(f"\n{request.get('description', '')}\n")

    # Stato finale
    final = graph.get_state(config).values
    status = final.get("status", "")
    if status == "completed":
        print("\n[SUCCESSO] Post approvato e Knowledge Graph aggiornato.")
    elif status == "discarded":
        print("\n[INFO] Operazione conclusa: bozza scartata.")
    else:
        print(f"\n[INFO] Processo concluso (stato: {status or 'n/d'}).")


def run_agent():
    _print_header()
    check_langsmith_setup()  # avvisa se il tracing LangSmith e' attivo o meno

    while True:
        user_input = input("\n[TU]: ").strip()
        if user_input.lower() in ("esci", "quit", "exit"):
            print("[INFO] Chiusura del Copilot. A presto!")
            break
        if not user_input:
            continue

        # Ogni richiesta in un thread separato (lo stato e' isolato per conversazione)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("\n[INFO] L'agente sta analizzando la richiesta...")

        try:
            result = graph.invoke({"user_input": user_input}, config)
        except Exception as e:
            print(f"\n[ERRORE] Esecuzione del grafo fallita: {e}")
            continue

        # FASE 0 - Gestione di eventuali CHIARIMENTI (anche piu' giri).
        # Finche' il grafo si ferma su un interrupt di tipo 'clarify_request',
        # dialoghiamo con l'utente e riprendiamo.
        request = _extract_interrupt(result)
        while request and _interrupt_action(request) == "clarify_request":
            result = _handle_clarification(request, config)
            request = _extract_interrupt(result)

        # Caso A: il grafo si e' fermato per la revisione della bozza (FASE 4 HITL)
        if request and _interrupt_action(request) == "review_post_draft":
            print("\n" + "=" * 56)
            print("  BOZZA PRONTA PER LA REVISIONE")
            print("=" * 56)
            print(f"\n{request.get('description', '')}\n")
            _run_review_loop(config)
        else:
            # Caso B: nessuna revisione (es. solo suggerimento di topic) -> mostra l'output
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                print(f"\n[RISPOSTA DELL'AGENTE]:\n{messages[-1].content}\n")
            else:
                print("\n[INFO] Operazione conclusa.")


if __name__ == "__main__":
    run_agent()
