"""
Popola il Knowledge Graph (Neo4j) con alcuni post-esempio.
Quando inizializzo da zero il progetto, il KG è vuoto, quindi il planner non avendo una gap-analysis, non c'è una
"memoria editoriale". Inserisco perciò degli articoli fittizi ma realistici per simulare uno storico editoriale
"""

from dotenv import load_dotenv
load_dotenv()

from .updater import update_kg_data

# Post-esempio che coprono vari argomenti. Ogni post è un dizionario con i campi topic, post_title, category, sources, claims, related_topics.
SEED_POSTS = [
    {
        "topic": "aerodinamica attiva",
        "post_title": "L'importanza dell'aerodinamica attiva nelle hypercar moderne",
        "category": "review",
        "sources": ["Rivista Aerodinamica e Design Automotive", "Test in galleria del vento 2023"],
        "claims": [
            "I flap mobili riducono il drag aerodinamico in rettilineo e aumentano il downforce in curva.",
            "I sistemi attivi possono ridurre i consumi di carburante fino al 15% alle alte velocità.",
            "L'ala posteriore con funzione di aerofreno dimezza gli spazi di arresto oltre i 200 km/h."
        ],
        "related_topics": ["prestazioni", "design funzionale", "hypercar"],
    },
    {
        "topic": "e-fuels carburanti sintetici",
        "post_title": "Carburanti Sintetici (e-fuels): la vera salvezza dei motori termici?",
        "category": "news",
        "sources": ["Report Europeo sulla transizione ecologica", "Dati di produzione impianto Haru Oni (Porsche)"],
        "claims": [
            "Gli e-fuels sono prodotti combinando idrogeno verde e CO2 catturata direttamente dall'atmosfera.",
            "Consentono una neutralità carbonica (Net Zero) pur mantenendo l'architettura dei motori endotermici.",
            "Attualmente il costo di produzione supera i 10 euro al litro, rendendoli poco accessibili per il mercato di massa."
        ],
        "related_topics": ["transizione ecologica", "motori termici", "innovazione sostenibile"],
    },
    {
        "topic": "pneumatici invernali e all season",
        "post_title": "Gomme 4 stagioni o invernali? Guida tecnica alla scelta della mescola perfetta",
        "category": "how_to",
        "sources": ["Test comparativo pneumatici 2024", "Manuale tecnico delle mescole termiche"],
        "claims": [
            "Gli pneumatici all-season perdono drasticamente efficacia in frenata con temperature inferiori ai 7°C.",
            "La mescola invernale pura contiene una percentuale maggiore di silice per evitare l'indurimento al gelo.",
            "Il battistrada invernale presenta lamelle a zig-zag progettate per 'aggrapparsi' alla neve."
        ],
        "related_topics": ["sicurezza stradale", "manutenzione periodica", "pneumatici"],
    },
    {
        "topic": "sospensioni predittive",
        "post_title": "Sospensioni intelligenti: come le auto moderne 'leggono' le buche",
        "category": "news",
        "sources": ["Scheda tecnica sistemi Magic Body Control", "Whitepaper Bosch sulle sospensioni attive"],
        "claims": [
            "Telecamere stereo scansionano le imperfezioni dell'asfalto fino a 15 metri davanti al veicolo.",
            "Gli attuatori idraulici compensano l'inclinazione della cassa in una frazione di secondo, neutralizzando il rollio.",
            "Richiedono una rete di bordo a 48 Volt per gestire i picchi di assorbimento energetico dei motorini elettrici."
        ],
        "related_topics": ["comfort di marcia", "tecnologia di bordo", "telai"],
    },
    {
        "topic": "restomod elettrici",
        "post_title": "Il fenomeno dei Restomod: dare un cuore a zero emissioni alle auto d'epoca",
        "category": "news",
        "sources": ["Intervista a preparatori specializzati in conversioni EV", "Trend report mercato auto storiche"],
        "claims": [
            "La conversione elettrica (retrofit) permette di circolare liberamente nelle ZTL con veicoli degli anni '60 e '70.",
            "L'installazione del pacco batterie altera la distribuzione dei pesi, modificando la dinamica di guida originale.",
            "In Italia la normativa sul retrofit elettrico (Decreto Retrofit) è stata semplificata per favorire l'omologazione."
        ],
        "related_topics": ["auto d'epoca", "retrofit elettrico", "tuning"],
    },
]


def main():
    print("Seed del Knowledge Graph")
    ok = 0
    for post in SEED_POSTS:
        try:
            result = update_kg_data(**post)
            print(f" Aggiunto post: '{post['post_title']}'")
            ok += 1
        except Exception as e:
            print(f" Errore nell'aggiunta del post '{post['post_title']}': {e}")
    print(f"\nInseriti {ok}/{len(SEED_POSTS)} post nel Knowledge Graph.")
if __name__ == "__main__":
    main()
