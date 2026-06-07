"""
Script che popola ChromaDB con i documenti locali della cartella "blog_sources"
Funziona similmente a seed.kg per KG, ma in questo caso mi serve per il RAG.
"""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .vectorstore import PERSIST_DIR, COLLECTION_NAME, embeddings

# Cartella che contiene i documenti privati del blogger (appunti, manuali, ecc)
SOURCE_DIR = "blog_sources"



# Metodo che prende tutti i file txt della cartellla
# li carica come documenti di LangChain
# li divide in chunk fissi da 500 caratteri
# più una sovrapposizione di 50 caratteri
# così non perdo informazioni tra due chunk
# trasformo ogni chunk in un vettore numerico 
# e lo salvo su ChromaDB.


def ingest_documents(source_dir: str = SOURCE_DIR):
    # 1. Caricamento di tutti i .txt della cartella
    file_paths = sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    if not file_paths:
        print(f"Nessun file .txt trovato in '{source_dir}/'. "
              f"Inserisci i tuoi appunti/manuali li' dentro e rilancia.")
        return

    docs = []
    for path in file_paths:
        docs.extend(TextLoader(path, encoding="utf-8").load())
    print(f"Caricati {len(docs)} documenti da {len(file_paths)} file in '{source_dir}/'.")

    # Recursive chunking + sliding window
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Dimensione fissa
        chunk_overlap=50,      # Finestra sovrapposta 
        separators=["\n\n", "\n", ".", " "],  
    )
    splits = text_splitter.split_documents(docs)

    # Embedding e salvataggio nel vector store locale
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Ingestione completata! Creati {len(splits)} chunk semantici in '{PERSIST_DIR}' "
          f"(collection '{COLLECTION_NAME}').")


if __name__ == "__main__":
    # Nel caso in cui eseguo lo script ma non ci sono ancora file txt dentro la cartella blog_sources
    # viene creato un file di esempio.
    os.makedirs(SOURCE_DIR, exist_ok=True)
    sample_path = os.path.join(SOURCE_DIR, "appunti_manutenzione_auto.txt")
    if not os.path.exists(sample_path):
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(
                "La manutenzione dei freni a disco va eseguita controllando lo spessore "
                "delle pastiglie: sotto i 3 mm vanno sostituite.\n\n"
                "Sulle auto elettriche la frenata rigenerativa riduce l'usura delle pastiglie, "
                "ma i dischi possono ossidarsi per il minor uso: va verificato periodicamente.\n\n"
                "Il liquido dei freni e' igroscopico e assorbe umidita': si consiglia la "
                "sostituzione ogni due anni per mantenere un punto di ebollizione sicuro."
            )
        print(f"Creato file di esempio: {sample_path}")

    ingest_documents()
