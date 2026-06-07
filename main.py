"""Entry point dell'AutomotiveBloggerAgent
Gestisce l'interfaccia a riga di comando (CLI) 
e, soprattutto, gestisce le interazioni "Human-In-The-Loop" (HITL), 
cioè i momenti in cui il grafo si mette in pausa per aspettare un input dall'utente.
"""

import uuid
from langgraph.types import Command
from agent import graph
from config import check_langsmith_setup


def _print_header():
    print("\n")
    print("AutomotiveBloggerAgent - Assistente editoriale automotive")
    print("\n")
    print("Esempi di richiesta:")
    print("\n")
    print("Scrivi un post sull'Alfa Romeo Giulia Quadrifoglio")
    print("Suggeriscimi argomenti per i prossimi post")
    print("Scrivi 'esci' per chiudere.\n")

# Metodo che gestisce la logica HITL da CLI. Quando metto in pausa il
# grafo di LangGraph, non torna il risultato finale, ma un dizionario speciale.

def _extract_interrupt(result):
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

# Metodo che estrae il motivo dell'interrupt (se è un chiarificazione per la fase 0 o per hitl fase 5)
def _interrupt_action(request) -> str:
    """Estrae il nome dell'azione da un request di interrupt."""
    try:
        return request.get("action_request", {}).get("action", "")
    except Exception:
        return ""


#Banale menu CLI per la revisione
def _ask_review_choice() -> dict:
    print("\n" + "-" * 56)
    print("Come vuoi procedere?")
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

# Gestisce la chiarificazione nello scoping.
def _handle_clarification(request, config):
    question = request.get("description", "Puoi chiarire meglio la tua richiesta?")
    print("\n" + "-" * 56)
    print("[CHIARIMENTO] L'agente ha bisogno di una precisazione:")
    print(f"  {question}")
    print("(Scrivi la tua risposta, oppure premi INVIO per procedere comunque.)")
    risposta = input("La tua risposta: ").strip()

    if not risposta:
        resume = {"type": "ignore"}  
    else:
        resume = {"type": "response", "args": risposta}

    return graph.invoke(Command(resume=resume), config)

# Gestisce la parte di revisione della bozza. Se ho scelto la modifica allora si riscrive l'articolo
# altrimenti il grafo non viene più fermato.
def _run_review_loop(config):
    while True:
        state = graph.get_state(config)
        if not state.next:
            # Il grafo e' arrivato alla fine (approvazione completata o flusso concluso)
            break

        scelta = _ask_review_choice()

        if scelta["type"] == "accept":
            print("\nApprovazione ricevuta: aggiornamento del Knowledge Graph.")
        elif scelta["type"] == "ignore":
            print("\nBozza scartata: nessun salvataggio.")
        else:
            print("\nApplico le modifiche e rigenero la bozza")

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
        print("\nPost approvato e Knowledge Graph aggiornato.")
    elif status == "discarded":
        print("\nOperazione conclusa: bozza scartata.")
    else:
        print(f"\nProcesso concluso (stato: {status or 'n/d'}).")


def run_agent():
    _print_header()
    check_langsmith_setup()  

    while True:
        user_input = input("\n[TU]: ").strip()
        if user_input.lower() in ("esci", "quit", "exit"):
            print("[INFO] Chiusura del Copilot. A presto!")
            break
        if not user_input:
            continue

        # Ogni richiesta in un thread separato (lo stato e' isolato per conversazione)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("\nL'agente sta analizzando la richiesta.")

        try:
            result = graph.invoke({"user_input": user_input}, config)
        except Exception as e:
            print(f"\nEsecuzione del grafo fallita: {e}")
            continue

        # Gestisco più chiarimenti (abbiamo un tetto massimo di 2)
        request = _extract_interrupt(result)
        while request and _interrupt_action(request) == "clarify_request":
            result = _handle_clarification(request, config)
            request = _extract_interrupt(result)

        # Caso in cui il grafo si ferma per la revisione
        if request and _interrupt_action(request) == "review_post_draft":
            print("\n" + "=" * 56)
            print("  BOZZA PRONTA PER LA REVISIONE")
            print("=" * 56)
            print(f"\n{request.get('description', '')}\n")
            _run_review_loop(config)
        else:
            # Quando non serve la revisione (solo suggerimenti richiesti e non post)
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                print(f"\nSuggerimenti dell'agente:\n{messages[-1].content}\n")
            else:
                print("\nOperazione conclusa.")


if __name__ == "__main__":
    run_agent()
