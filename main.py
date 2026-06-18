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
    print("L'agente ha bisogno di una precisazione:")
    print(f"  {question}")
    print("(Scrivi la tua risposta, oppure premi INVIO per procedere comunque.)")
    risposta = input("La tua risposta: ").strip()

    if not risposta:
        resume = {"type": "ignore"}  
    else:
        resume = {"type": "response", "args": risposta}

    return graph.invoke(Command(resume=resume), config)

# Gestisce la fase editoriale HITL. Mostra le proposte pianificate e raccoglie
# la decisione dell'utente in linguaggio naturale (quali scrivere, quali modificare con
# istruzioni, quali scartare, se proporne di nuove). INVIO vuoto o "annulla" allora annulla tutto.
def _handle_editorial_review(request, config):
    print("\n" + "=" * 56)
    print("Revisione del piano editoriale: \n")
    print("=" * 56)
    print(f"\n{request.get('description', '')}\n")
    risposta = input("La tua scelta (INVIO o 'annulla' per annullare): ").strip()
    # Riconosco le varianti naturali di annullamento (dal test: "annullare" non veniva capito).
    if not risposta or risposta.lower() in (
        "annulla", "annullare", "annulla tutto", "annulla pure", "cancella", "cancella tutto",
        "esci", "lascia stare", "niente", "no grazie",
    ):
        resume = {"type": "ignore"}
    else:
        resume = {"type": "response", "args": risposta}
    return graph.invoke(Command(resume=resume), config)


# Gestisce la revisione di una bozza (HITL fase 4). Mostra la bozza, chiede la scelta
# (approva/modifica/scarta) e riprende il grafo iniettando la risposta.
def _handle_review_draft(request, config):
    print("\n" + "=" * 56)
    print("Bozza pronta per essere revisionata: \n")
    print("=" * 56)
    print(f"\n{request.get('description', '')}\n")
    scelta = _ask_review_choice()
    if scelta["type"] == "accept":
        print("\nApprovazione ricevuta: aggiornamento del Knowledge Graph.")
    elif scelta["type"] == "ignore":
        print("\nBozza scartata: nessun salvataggio.")
    else:
        print("\nApplico le modifiche e rigenero la bozza.")
    return graph.invoke(Command(resume=scelta), config)


# Gestisce il gate 'prossimo post' (HITL del ciclo multi-post). Mostra i post selezionati
# rimasti e chiede se continuare (e con quale) o fermarsi. INVIO vuoto / "fermati" si ferma;
# "continua" prende il primo; un numero o un tema scelgono quale scrivere.
def _handle_continue(request, config):
    print("\n" + "-" * 56)
    print("  Prossimo post: \n")
    print(f"\n{request.get('description', '')}\n")
    risposta = input("Continua con quale? (Invio o 'fermati' per fermarti): ").strip()
    if not risposta or risposta.lower() in ("fermati", "basta", "stop", "no"):
        resume = {"type": "ignore"}
    elif risposta.lower() in ("continua", "si", "sì", "vai", "prosegui"):
        resume = {"type": "accept"}
    else:
        resume = {"type": "response", "args": risposta}
    return graph.invoke(Command(resume=resume), config)


# Gestisce il gate dei suggerimenti: mostra le proposte (in sospeso, calendario, RSS)
# e chiede se l'utente vuole scrivere uno di quei temi. INVIO/no -> chiude e basta;
# altrimenti la scelta viene interpretata dal nodo e si riparte verso la scrittura.
def _handle_choose_suggestion(request, config):
    print("\n" + "=" * 56)
    print("  Suggerimenti dell'agente: \n")
    print("=" * 56)
    print(f"\n{request.get('description', '')}\n")
    print("Vuoi scrivere uno di questi temi? Indica quale (es. 'la proposta 1',")
    print("'il calendario 2', 'la notizia 3', oppure descrivi il tema).")
    risposta = input("La tua scelta (INVIO o 'no' per chiudere): ").strip()
    if not risposta or risposta.lower() in ("no", "no grazie", "niente", "nulla",
                                            "lascia stare", "esci", "chiudi", "annulla"):
        resume = {"type": "ignore"}
    else:
        resume = {"type": "response", "args": risposta}
    return graph.invoke(Command(resume=resume), config)


# Cuore della gestione HITL: finche' il grafo si ferma su un interrupt, lo smista
# all'handler giusto in base all'azione. Gestisce catene di pause concatenate
# (chiarimento -> gate editoriale -> revisione bozza -> prossimo post).
def _dispatch_interrupts(result, config):
    request = _extract_interrupt(result)
    while request:
        action = _interrupt_action(request)
        if action == "clarify_request":
            result = _handle_clarification(request, config)
        elif action == "review_editorial_plan":
            result = _handle_editorial_review(request, config)
        elif action == "review_post_draft":
            result = _handle_review_draft(request, config)
        elif action == "continue_writing":
            result = _handle_continue(request, config)
        elif action == "choose_suggestion":
            result = _handle_choose_suggestion(request, config)
        else:
            # Azione di interrupt sconosciuta: per sicurezza esco dal loop.
            break
        request = _extract_interrupt(result)
    return result


def run_agent():
    _print_header()
    check_langsmith_setup()  

    while True:
        user_input = input("\n[TU]: ").strip()
        if user_input.lower() in ("esci", "quit", "exit"):
            print("Chiusura dell'AutomotiveBloggerAgent. A presto!")
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

        # Smisto tutte le pause HITL concatenate: chiarimenti, gate editoriale,
        # revisione delle bozze e gate 'prossimo post' del ciclo multi-post.
        try:
            result = _dispatch_interrupts(result, config)
        except Exception as e:
            print(f"\nErrore durante la gestione dell'interazione: {e}")
            continue

        # Stato finale + eventuali messaggi (es. solo suggerimenti, senza scrittura).
        final = graph.get_state(config).values
        status = final.get("status", "") if isinstance(final, dict) else ""
        messages = final.get("messages", []) if isinstance(final, dict) else []

        if status in ("completed", "completed_all"):
            print("\nProcesso concluso: post pubblicati e Knowledge Graph aggiornato.")
        elif status == "stopped_with_pending":
            print("\nOk, mi fermo. I post non scritti restano come proposte recuperabili nel KG.")
        elif status == "discarded":
            print("\nBozza scartata: nessun salvataggio.")
        elif status == "planning_cancelled":
            print("\nPianificazione annullata: nessun post scritto.")
        elif status == "topics_suggested":
            # I suggerimenti sono gia' stati mostrati dal gate HITL, chiudo e basta.
            print("\nOk! Quando vuoi scrivere uno di questi temi, chiedimelo pure.")
        else:
            # Fallback generico.
            if messages:
                print(f"\n{messages[-1].content}\n")
            else:
                print(f"\nProcesso concluso (stato: {status or 'n/d'}).")


if __name__ == "__main__":
    run_agent()
