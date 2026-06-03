import os
from dotenv import load_dotenv
from langsmith import Client

# Carichiamo le variabili d'ambiente (serve LANGSMITH_API_KEY per autenticarsi a LangSmith)
load_dotenv()

client = Client()

dataset_name = "Blogger_Test_Dataset_v2"
dataset_description = (
    "Dataset esteso (15 prompt) per la valutazione FINALE del Blogger Copilot sul sistema "
    "completo. Copre tutte le casistiche: confronti tra veicoli (compare_vehicles + modello "
    "fine-tuned), post tecnici (RAG locale), attualita' (web search + guardrail fonti), "
    "recensioni (fetch_vehicle_specs), suggerimento argomenti (gap-analysis del KG) e casi "
    "difficili. Affianca il dataset iniziale a 5 prompt per misurare il miglioramento."
)

# 15 prompt diversificati: ognuno stressa un ramo/tool diverso del grafo.
# La nota a fianco indica il comportamento atteso (per analisi dei risultati).
test_cases = [
    # --- Confronti tra veicoli (compare_vehicles + fine-tuned) ---
    ("Scrivi un post che confronta la Volkswagen Golf 2024 e la Toyota Corolla 2024",
     "Usa compare_vehicles (modello fine-tuned) e produce un confronto strutturato con fonti."),
    ("Confronta la Ducati Panigale V2 e la Yamaha R7 in un post",
     "Confronto tra due moto: compare_vehicles (fine-tuned), verdetto per categorie."),
    ("Scrivi un post che confronta motori termici ed elettrici",
     "Confronto concettuale: RAG locale (documento sul confronto) e/o web, post discorsivo."),
    # --- Post tecnici (RAG locale: temi presenti nei documenti) ---
    ("Scrivi un post sulla manutenzione dei freni a disco",
     "RAG locale rilevante (manutenzione_freni_disco): post ancorato ai documenti."),
    ("Scrivi un post sulle batterie allo stato solido",
     "RAG locale (batterie_stato_solido) + eventuale web: grounding alto atteso."),
    ("Scrivi un post sui sistemi ADAS e la loro calibrazione",
     "RAG locale (sistemi_adas_e_calibrazione): contenuto tecnico ancorato."),
    ("Scrivi un post sul funzionamento del cambio a doppia frizione",
     "RAG locale (cambi_doppia_frizione): spiegazione tecnica fondata sulle fonti."),
    ("Scrivi un post sulla frenata rigenerativa nelle auto elettriche",
     "RAG locale (frenata_rigenerativa_elettriche): post tecnico ancorato."),
    ("Scrivi un post sui sistemi di post-trattamento dei gas di scarico",
     "RAG locale (sistemi_post_trattamento): contenuto tecnico fondato."),
    ("Parlami delle reti di comunicazione di bordo CAN bus",
     "RAG locale (reti_di_comunicazione_di_bordo): post tecnico; testa anche il routing 'parlami'."),
    # --- Attualita' / novita' (web search + guardrail verifica fonti) ---
    ("Scrivi un post sulle ultime novità delle auto elettriche 2026",
     "Tema di attualita': web search; il guardrail forza una ricerca se mancano fonti."),
    ("Scrivi un post sui principali saloni automobilistici europei",
     "Attualita'/eventi: web search; rischio che il modello scriva a memoria (failure case)."),
    # --- Recensioni modello singolo (fetch_vehicle_specs) ---
    ("Scrivi una recensione tecnica della Fiat 600 elettrica",
     "Modello specifico: fetch_vehicle_specs; se la fonte e' debole, il guardrail interviene."),
    ("Scrivi un post sulla nuova Aprilia Tuono 457",
     "Modello recente: fetch_vehicle_specs probabilmente fallisce -> guardrail web (failure case)."),
    # --- Suggerimento argomenti (ramo suggest_topics + gap-analysis del KG) ---
    ("Suggeriscimi argomenti per i prossimi post del blog",
     "Ramo suggerimenti: gap-analysis sul KG, propone temi non ancora trattati."),
]


def get_or_create_dataset():
    """Restituisce il dataset, creandolo se non esiste (idempotente: rilanciabile)."""
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        print(f"Dataset '{dataset_name}' gia' esistente: lo riuso (id={existing[0].id}).")
        return existing[0]
    print(f"Creazione del dataset '{dataset_name}' in corso...")
    return client.create_dataset(dataset_name=dataset_name, description=dataset_description)


def main():
    dataset = get_or_create_dataset()

    # Evitiamo di duplicare gli esempi se il dataset e' gia' popolato
    already = list(client.list_examples(dataset_id=dataset.id))
    if already:
        print(f"Il dataset contiene gia' {len(already)} esempi: non ne aggiungo di nuovi.")
        return

    for prompt, expected in test_cases:
        client.create_example(
            inputs={"user_input": prompt},
            outputs={"expected_behavior": expected},
            dataset_id=dataset.id,
        )
    print(f"Dataset '{dataset_name}' popolato con {len(test_cases)} esempi. "
          f"Visibile sulla dashboard di LangSmith.")


if __name__ == "__main__":
    main()
