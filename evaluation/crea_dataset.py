import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
client = Client()

# File per creare il dataset da usare per l'evaluation.
# Avviando questo script si crea un nuovo dataset con i prompt di test.
# Successivamente, usando evaluate_blog.py, si userà questo dataset per la valutazione.


dataset_name = "AutomotiveBloggerAgent Dataset 1"
dataset_description = "Dataset breve per la valutazione di AutomotiveBloggerAgent"

# Prompt per vari scenari del progetto (post, news, recensione, KG, eventi)
test_inputs = [
    "Scrivi un post dettagliato sulle differenze tra auto elettriche e auto a idrogeno.",
    "Quali sono le ultime notizie rilevanti nel mondo delle moto elettriche?",
    "Scrivi una recensione tecnica del nuovo SUV Alfa Romeo Milano (Junior) 2024.",
    "Oggi vorrei scrivere un post sulle batterie allo stato solido, che ne pensi?",  
    "Fammi una lista dei prossimi grandi eventi o saloni automobilistici in Europa.",
]


def get_or_create_dataset():
    """
    Restituisce il dataset, creandolo se non esiste.
    Evita l'errore di 'create_dataset' quando il dataset e' gia' presente
    """
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        print(f"Dataset '{dataset_name}' gia' esistente: lo riuso (id={existing[0].id}).")
        return existing[0]
    print(f"Creazione del dataset '{dataset_name}' in corso")
    return client.create_dataset(dataset_name=dataset_name, description=dataset_description)


def main():
    dataset = get_or_create_dataset()

    # Evitiamo di duplicare gli esempi se il dataset e' gia' popolato
    already = list(client.list_examples(dataset_id=dataset.id))
    if already:
        print(f"Il dataset contiene gia' {len(already)} esempi.")
        return

    for prompt in test_inputs:
        client.create_example(
            inputs={"user_input": prompt},
            outputs={"expected_behavior": "Generazione post o suggerimenti ancorati a fonti reali."},
            dataset_id=dataset.id,
        )
    print(f"Dataset popolato con {len(test_inputs)} esempi e visibile sulla dashboard di LangSmith.")


if __name__ == "__main__":
    main()