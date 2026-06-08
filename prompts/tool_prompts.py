
# Modulo che contiene le descrizioni di ogni tool disponibile per l'agente, 
# inlcusi i prompt e le regole che definiscono come usare questi strumenti. 


# Prompt per la ricerca web attraverso Tavily e MCP. Deve essere
# usato per trovare fonti ed informazioni reali/attuali.

WEB_SEARCH_PROMPT = """
Scopo: Invia una richiesta al Server MCP (Microservizio) per effettuare una ricerca web profonda tramite Tavily.
Quando usarlo: USA QUESTO TOOL quando il tema riguarda ATTUALITA', novita', notizie recenti, nuovi modelli, dati di mercato o eventi (es. 'novita' 2026', 'ultime notizie', 'nuovo modello X'). E' la fonte giusta per informazioni che cambiano nel tempo e non sono negli appunti locali.
Azione: Restituisce un riassunto tecnico e le fonti originali. REGOLA ASSOLUTA: Se integri questi dati nel post, devi obbligatoriamente citare la fonte esterna nel contenuto generato, per dimostrare che il testo è supportato da documenti reali.
"""

# Prompt per il RAG, viene usato per recuperare
# appunti/informazioni tecniche dai documenti locali.

RAG_RETRIEVAL_PROMPT = """
Scopo: Interroga il database vettoriale locale (ChromaDB) contenente gli appunti privati del blogger, manuali tecnici e vecchi articoli.
Quando usarlo: Usa questo strumento quando devi scrivere un articolo e hai bisogno di recuperare informazioni tecniche specifiche salvate in locale o dettagli da appunti personali.
Azione: Fornisce i frammenti di testo (chunk) dalla memoria locale. REGOLE ASSOLUTE: Non inventare dati. Se usi informazioni provenienti da questo tool nella tua bozza, DEVI esplicitamente inserire la citazione o il riferimento alla fonte nel testo finale.
"""

# Prompt per la cronologia di un topic specifico all'interno del Knowledge Graph.
# Serve a far capire al modello se ha già parlato di un determinato argomento, 
# in modo da non creare ripetizioni di contenuti.

KG_QUERY_PROMPT = """
Scopo: Legge il Knowledge Graph (Neo4j) del blog per la cronologia di UN argomento specifico.
Quando usarlo: 
1. Nella fase di PIANIFICAZIONE (Topic suggestion): per verificare se un argomento specifico è già stato trattato, evitando doppioni.
2. Nella fase di STESURA (Post drafting): per un controllo rapido sui post passati legati a quel topic.
Azione: Restituisce la lista dei post già pubblicati su quel topic (con data) e i topic correlati.

Nota per la scelta del tool KG corretto:
- Per la cronologia di UN topic preciso -> usa questo tool ('query_knowledge_graph').
- Per la panoramica di TUTTI i topic e per trovare i GAP di copertura -> usa 'list_blog_topics'.
- Per il contesto COMPLETO (post, claim, fonti, correlati) utile in stesura -> usa 'get_editorial_context'.
"""


# Prompt per la panoramica di tutti i topic
# Serve a far capire al modello i gap di copertura
# e proporre post diversi.

KG_TOPICS_OVERVIEW_PROMPT = """
Scopo: Restituisce la panoramica COMPLETA della copertura editoriale dal Knowledge Graph:
tutti i topic con numero di post e data dell'ultimo post, ordinati dal più trascurato.
Quando usarlo: Nella fase di PIANIFICAZIONE/brainstorming, per individuare i GAP di copertura
e gli argomenti non trattati di recente, così da proporre post nuovi e diversificati.
Azione: Non richiede parametri. I topic 'MAI trattato' sono i gap prioritari.
"""


# Prompt per il contesto editoriale completo di un topic
# Alimenta il K-RAG con i claim da non contraddire, fonti usate
# e possibili citazioni cross post.

KG_CONTEXT_PROMPT = """
Scopo: Recupera il contesto editoriale di un topic dal Knowledge Graph: post esistenti,
claim già affermati, fonti già usate e topic correlati.
Quando usarlo: Nella fase di STESURA, per garantire coerenza con i contenuti passati,
evitare di contraddire claim già pubblicati e creare collegamenti interni (cross-link).
Input richiesto: topic (str) - l'argomento dell'articolo che stai scrivendo.
"""

# Prompt di scrittura nel KG (il più importante dei 4). 
# Serve a dire esattamente all'agente che tipi di parametri
# deve estrarre dal post per la pubblicazione su Neo4J e soprattutto
# non deve mai essere chiamato prima dell'approvazione finale dell'utente.

KG_UPDATE_PROMPT = """
Scopo: Scrive un nuovo nodo e una relazione nel Knowledge Graph (Neo4j).
Quando usarlo: ATTENZIONE - Usa questo tool ESCLUSIVAMENTE alla fine del processo, SOLO DOPO che l'utente ha approvato esplicitamente la bozza finale dell'articolo. 
Azione: Registra il nuovo articolo nel database per la memoria futura. Non chiamarlo mai durante la fase di stesura o ricerca.

Devi estrarre dal contesto e fornire i seguenti parametri obbligatori:
- topic (str): L'argomento generale o il tema principale del post.
- post_title (str): Il titolo definitivo dell'articolo appena approvato.
- category (str): La macro-categoria di appartenenza del post.
- sources (list[str]): Una lista di URL o nomi delle fonti esterne/RAG che hai utilizzato per redigere l'articolo.
- claims (list[str]): Una lista di 3 o 4 affermazioni o concetti chiave estratti dall'articolo.
- related_topics (list[str]): Una lista di argomenti correlati al topic principale, utili per espandere il grafo.
"""


# Prompt per il tool di generazione immagini di copertina
# viene invocato all'approvazione del post (non ha senso prima)


IMAGE_GENERATOR_PROMPT = """
Scopo: Genera fisicamente un'immagine fotorealistica di copertina per l'articolo e la salva sul computer (Cloudflare FLUX, con fallback Pollinations AI).
Input richiesto: Fornisci un prompt testuale MOLTO DESCRITTIVO in LINGUA INGLESE che descriva il SOGGETTO e la SCENA in dettaglio (veicolo, ambiente, luce, atmosfera), es. "A hyper-realistic cinematic shot of a red Ferrari Roma driving on a wet mountain road during twilight". NON chiedere testo, scritte, titoli o loghi nell'immagine: le direttive di stile e i divieti di testo vengono aggiunti automaticamente.
Quando usarlo: Quando l'articolo è pronto o quando l'utente chiede esplicitamente una copertina per il post.
"""

# Prompt per l'analisi SEO (Search Engine Optimization) 
# viene invocato all'approvazione del post, valuta
# la qualità di lettura del post, difficoltà grammaticale, 
# leggibilità del testo in italiano, per un blog tecnico automotive, 40-60 e' il range adeguato).

SEO_ANALYSIS_PROMPT = """
Scopo: Calcola due metriche SEO sulla bozza dell'articolo, in modo deterministico (formula matematica, nessuna opinione):
  1. DENSITA' KEYWORD: quante volte la parola chiave principale compare nel testo (ideale: 0.5%-2.5%).
  2. LEGGIBILITA' (indice Gulpease): quanto e' leggibile il testo in italiano (per un blog tecnico automotive, 40-60 e' il range adeguato).
Input: il testo completo della bozza + la keyword (parola chiave) su cui posizionare l'articolo.
Quando usarlo: SOLO dopo che la bozza e' stata scritta; misura oggettivamente la qualita' SEO del post appena generato.
"""


# Prompt per tool che recupera i titoli e le anteprime delle ULTIME NOTIZIE automotive 
# dai feed RSS di testate specializzate (Motor1, Autoblog ecc.). Viene invocato
# solo quando l'utente chiede idee o argomenti per nuovi post e quando sta
# scrivendo un articolo su TENDENZE O NOVITA' di mercato e vuole notizie fresche
# dalle testate. Non usarlo se il tema e' tecnico/enciclopedico (freni, batterie, ADAS):
# per quelli usa retrieve_local_documents o fetch_vehicle_specs.

TREND_ANALYSIS_PROMPT = """
Scopo: Recupera i titoli e le anteprime delle ULTIME NOTIZIE automotive dai feed RSS di testate specializzate (Motor1, Autoblog ecc.).
Input: query (stringa) — l'argomento o il segmento su cui cercare tendenze (es. 'auto elettriche', 'SUV', 'moto naked').
Quando usarlo: USA QUESTO TOOL in due situazioni precise:
  1. Quando l'utente chiede idee o argomenti per nuovi post ('di cosa potremmo parlare?', 'suggeriscimi un tema caldo').
  2. Quando stai scrivendo un articolo su TENDENZE O NOVITA' di mercato e vuoi notizie fresche dalle testate.
NON usarlo se il tema e' tecnico/enciclopedico (freni, batterie, ADAS): per quelli usa retrieve_local_documents o fetch_vehicle_specs.
Azione: Restituisce una lista di titoli recenti dal feed RSS. Usali per proporre idee basate sulle notizie del momento.
"""


# Prompt per tool che recupera i dati tecnici specifici di un veicolo.
# Non serve per confrontare tra loro i veicoli, serve solo a recuperare i dati di
# un veicolo, per es. kW, CV, peso, cilindrata ecc. Usalo quando devi
# stendere un articolo o quando devi integrare i dati tecnici.

VEHICLE_SPECS_PROMPT = """
Scopo: Recupera la scheda tecnica e il contesto enciclopedico di UN SINGOLO veicolo, combinando due fonti: API Ninja (dati tecnici strutturati: cilindrata, potenza, consumi, ecc.) e Wikipedia Italia (storia, background, descrizione del modello).
Input: car_model (stringa) — il nome completo del veicolo richiesto dall'utente. Passa SEMPRE il veicolo di cui si sta parlando nella conversazione, non un esempio.
Quando usarlo: USA QUESTO TOOL quando devi recuperare dati tecnici, cilindrate, anni di produzione, scheda tecnica o aneddoti storici di UN modello specifico. Serve per articoli monografici (recensioni) o per integrare dati tecnici nella stesura.
ATTENZIONE: per CONFRONTARE due veicoli NON usare questo tool due volte — usa 'compare_vehicles' che fa il confronto completo in un'unica chiamata.
Azione: Restituisce un riassunto strutturato che unisce le specifiche tecniche (da API Ninja, quando disponibili) e la storia del modello (da Wikipedia).
"""


# Prompt per la sintesi delle fonti estratte dalla ricerca web
MCP_SUMMARIZER_PROMPT = """
Sei un analista esperto di automobili e motori. Leggi i seguenti estratti di articoli 
web grezzi e scrivi un riassunto tecnico e dettagliato. 
Ignora le pubblicita' e le frasi inutili. Concentrati esclusivamente su: prezzi, 
specifiche dei motori, cavalli, design, tecnologia e date di uscita.

Testi grezzi dal web:
{full_text}

Scrivi solo il riassunto tecnico:
"""

# Prompt per il tool di ricerca veicoli + confronto delle specifiche
# questa coppia di prompt vengono usate per la comparazione dei veicoli.
# i dati grezzi vengono estratti con Ministral. Il modello deve seguire
# regole specifiche di formattazione dei dati, necessarie affinché il modello
# fine tuned funzioni come è stato allenato. Infatti llama 3.2 1B è stato allenato
# con un dataset di elementi formati da 150/200 token con una finestra massima di 2048.
# Di conseguenza se passassi roba troppo discorsiva, il giudice fallirebbe.

VEHICLE_RESEARCH_PROMPT = """Sei un analista dati del settore automotive.
Dai dati grezzi qui sotto, scrivi un profilo tecnico BREVE e FATTUALE del veicolo richiesto.

VEICOLO RICHIESTO: {query_base}

REGOLE RIGIDE:
1. UN SOLO PARAGRAFO discorsivo, MASSIMO 60-80 parole. NON usare titoli, sezioni, elenchi
   puntati, simboli Markdown (#, *, -) o sotto-paragrafi: solo testo discorsivo continuo.
2. Includi solo i dati chiave essenziali: motore e potenza (CV), consumi, prezzo, sicurezza
   (stelle Euro NCAP), dotazione principale. Tralascia tutto il resto.
3. Estrai solo numeri e fatti presenti nel testo grezzo. Se un dato manca, ometterlo
   (NON scrivere "non disponibile", NON inventare).
4. Niente aggettivi enfatici, niente note, niente criticita' o sezioni aggiuntive.

Esempio del livello di sintesi richiesto:
"Motore 1.0 3 cilindri mild-hybrid da 70 CV. Consumo medio dichiarato 4.6 l/100km. Prezzo da
17.500 EUR. 4 stelle Euro NCAP, frenata automatica assente nella base. Infotainment 7 pollici
con Apple CarPlay e clima manuale."

TESTO GREZZO RECUPERATO:
{testo_grezzo}

Scrivi il profilo tecnico breve (un paragrafo, max 60-80 parole):"""

# Prompt del modellino da 1B con fine tuning. Ritorna una struttura precisa (senza tabelle e cose strane).
# La comparazione è articolata in 4 punti + un verdetto finale.
# In ogni punti vi è una breve spiegazione del perché quel modelllo di veicolo vince (o perde/pareggia)
# con un giudizio finale

TINY_JUDGE_SYSTEM_PROMPT = """Sei un ingegnere specializzato in comparazioni automobilistiche e motociclistiche.
Il tuo compito è analizzare i profili tecnici di due veicoli e decretare un vincitore per categorie specifiche, basandoti ESCLUSIVAMENTE sui dati forniti nei tag XML.

STRUTTURA DI OUTPUT OBBLIGATORIA:
Rispondi usando esattamente questo formato in Markdown:

### Analisi Comparativa: [Nome Veicolo 1] vs [Nome Veicolo 2]

**1. Prestazioni e Motore**
*Vincitore:* [Nome Veicolo o Pareggio]
*Motivazione:* [Spiegazione tecnica in 1-2 frasi basata sui dati]

**2. Consumi e Costi Operativi**
*Vincitore:* [Nome Veicolo o Pareggio]
*Motivazione:* [Spiegazione tecnica in 1-2 frasi basata sui dati]

**3. Sicurezza e Affidabilità**
*Vincitore:* [Nome Veicolo o Pareggio]
*Motivazione:* [Spiegazione tecnica in 1-2 frasi basata sui difetti o test NCAP]

**4. Comfort e Tecnologia**
*Vincitore:* [Nome Veicolo o Pareggio]
*Motivazione:* [Spiegazione tecnica in 1-2 frasi basata sulle dotazioni]

**Verdetto Finale**
[Un paragrafo conclusivo di 3 righe che riassume a chi è destinato il primo veicolo e a chi il secondo].

VINCOLI:
- Non aggiungere altre categorie.
- Non inventare specifiche tecniche non menzionate nei dati in ingresso."""


