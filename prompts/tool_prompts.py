# tool_prompts.py

WEB_SEARCH_PROMPT = """
Scopo: Invia una richiesta al Server MCP (Microservizio) per effettuare una ricerca web profonda tramite Tavily.
Quando usarlo: USA QUESTO TOOL quando il tema riguarda ATTUALITA', novita', notizie recenti, nuovi modelli, dati di mercato o eventi (es. 'novita' 2026', 'ultime notizie', 'nuovo modello X'). E' la fonte giusta per informazioni che cambiano nel tempo e non sono negli appunti locali.
Azione: Restituisce un riassunto tecnico e le fonti originali. REGOLA ASSOLUTA: Se integri questi dati nel post, devi obbligatoriamente citare la fonte esterna nel contenuto generato, per dimostrare che il testo è supportato da documenti reali.
"""

RAG_RETRIEVAL_PROMPT = """
Scopo: Interroga il database vettoriale locale (ChromaDB) contenente gli appunti privati del blogger, manuali tecnici e vecchi articoli.
Quando usarlo: Usa questo strumento quando devi scrivere un articolo e hai bisogno di recuperare informazioni tecniche specifiche salvate in locale o dettagli da appunti personali.
Azione: Fornisce i frammenti di testo (chunk) dalla memoria locale. REGOLE ASSOLUTE: Non inventare dati. Se usi informazioni provenienti da questo tool nella tua bozza, DEVI esplicitamente inserire la citazione o il riferimento alla fonte nel testo finale.
"""

# ==========================================
# TOOL DEL KNOWLEDGE GRAPH (3 letture + 1 scrittura)
# ==========================================

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

KG_TOPICS_OVERVIEW_PROMPT = """
Scopo: Restituisce la panoramica COMPLETA della copertura editoriale dal Knowledge Graph:
tutti i topic con numero di post e data dell'ultimo post, ordinati dal più trascurato.
Quando usarlo: Nella fase di PIANIFICAZIONE/brainstorming, per individuare i GAP di copertura
e gli argomenti non trattati di recente, così da proporre post nuovi e diversificati.
Azione: Non richiede parametri. I topic 'MAI trattato' sono i gap prioritari.
"""

KG_CONTEXT_PROMPT = """
Scopo: Recupera il contesto editoriale di un topic dal Knowledge Graph: post esistenti,
claim già affermati, fonti già usate e topic correlati.
Quando usarlo: Nella fase di STESURA, per garantire coerenza con i contenuti passati,
evitare di contraddire claim già pubblicati e creare collegamenti interni (cross-link).
Input richiesto: topic (str) - l'argomento dell'articolo che stai scrivendo.
"""

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

IMAGE_GENERATOR_PROMPT = """
Scopo: Genera fisicamente un'immagine fotorealistica di copertina per l'articolo e la salva sul computer (Cloudflare FLUX, con fallback Pollinations AI).
Input richiesto: Fornisci un prompt testuale MOLTO DESCRITTIVO in LINGUA INGLESE che descriva il SOGGETTO e la SCENA in dettaglio (veicolo, ambiente, luce, atmosfera), es. "A hyper-realistic cinematic shot of a red Ferrari Roma driving on a wet mountain road during twilight". NON chiedere testo, scritte, titoli o loghi nell'immagine: le direttive di stile e i divieti di testo vengono aggiunti automaticamente.
Quando usarlo: Quando l'articolo è pronto o quando l'utente chiede esplicitamente una copertina per il post.
"""

SEO_ANALYSIS_PROMPT = """
Scopo: Calcola due metriche SEO sulla bozza dell'articolo, in modo deterministico (formula matematica, nessuna opinione):
  1. DENSITA' KEYWORD: quante volte la parola chiave principale compare nel testo (ideale: 0.5%-2.5%).
  2. LEGGIBILITA' (indice Gulpease): quanto e' leggibile il testo in italiano (per un blog tecnico automotive, 40-60 e' il range adeguato).
Input: il testo completo della bozza + la keyword (parola chiave) su cui posizionare l'articolo.
Quando usarlo: SOLO dopo che la bozza e' stata scritta; misura oggettivamente la qualita' SEO del post appena generato.
"""

TREND_ANALYSIS_PROMPT = """
Scopo: Recupera i titoli e le anteprime delle ULTIME NOTIZIE automotive dai feed RSS di testate specializzate (Motor1, Autoblog ecc.).
Input: query (stringa) — l'argomento o il segmento su cui cercare tendenze (es. 'auto elettriche', 'SUV', 'moto naked').
Quando usarlo: USA QUESTO TOOL in due situazioni precise:
  1. Quando l'utente chiede idee o argomenti per nuovi post ('di cosa potremmo parlare?', 'suggeriscimi un tema caldo').
  2. Quando stai scrivendo un articolo su TENDENZE O NOVITA' di mercato e vuoi notizie fresche dalle testate.
NON usarlo se il tema e' tecnico/enciclopedico (freni, batterie, ADAS): per quelli usa retrieve_local_documents o fetch_vehicle_specs.
Azione: Restituisce una lista di titoli recenti dal feed RSS. Usali per proporre idee basate sulle notizie del momento.
"""

VEHICLE_SPECS_PROMPT = """
Scopo: Interroga Wikipedia Italia per estrarre la scheda tecnica completa, la storia e il background enciclopedico di UN SINGOLO veicolo.
Input: car_model (stringa) — il nome completo del veicolo (es. 'Alfa Romeo Giulia', 'Ducati Panigale V4').
Quando usarlo: USA QUESTO TOOL quando devi recuperare dati storici, cilindrate, anni di produzione, scheda tecnica o aneddoti di UN modello specifico. Serve per articoli monografici (recensioni) o per integrare dati tecnici nella stesura.
ATTENZIONE: per CONFRONTARE due veicoli NON usare questo tool due volte — usa 'compare_vehicles' che fa il confronto completo in un'unica chiamata.
Azione: Restituisce un riassunto strutturato con le specifiche estratte da Wikipedia.
"""

MCP_SUMMARIZER_PROMPT = """
Sei un analista esperto di automobili e motori. Leggi i seguenti estratti di articoli 
web grezzi e scrivi un riassunto tecnico e dettagliato. 
Ignora le pubblicita' e le frasi inutili. Concentrati esclusivamente su: prezzi, 
specifiche dei motori, cavalli, design, tecnologia e date di uscita.

Testi grezzi dal web:
{full_text}

Scrivi solo il riassunto tecnico:
"""

# ==========================================
# PROMPT PER IL TOOL DI COMPARAZIONE VEICOLI
# ==========================================

# 1. Prompt per Phi-4 (Ricerca e Sintesi)
PHI4_VEHICLE_RESEARCH_PROMPT = """Sei un analista dati specializzato nel settore automotive.
Il tuo obiettivo è leggere i dati grezzi estratti dal web e compilare un profilo tecnico denso e puramente fattuale del veicolo richiesto.

VEICOLO RICHIESTO: {query_base}

REGOLE RIGIDE:
1. Ignora qualsiasi testo promozionale, pubblicitario o non correlato al veicolo.
2. Estrai esclusivamente numeri e fatti dimostrabili (CV, kW, dimensioni, consumi dichiarati e reali, tempi di ricarica, difetti riportati, stelle Euro NCAP).
3. Non usare aggettivi enfatici.
4. Se un dato non è presente nel testo grezzo, non inventarlo.

TESTO GREZZO RECUPERATO:
{testo_grezzo}

Scrivi il profilo tecnico in modo conciso e strutturato:"""

# 2. Prompt per LLaMA 3.2 1B (Il Giudice Fine-Tuned)
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

**🏆 Verdetto Finale**
[Un paragrafo conclusivo di 3 righe che riassume a chi è destinato il primo veicolo e a chi il secondo].

VINCOLI:
- Non aggiungere altre categorie.
- Non inventare specifiche tecniche non menzionate nei dati in ingresso."""

THINK_TOOL_PROMPT = """Strumento di RIFLESSIONE strategica (Thought esplicito del ciclo ReAct).
Usalo per fermarti ad analizzare i risultati ottenuti e pianificare il prossimo passo,
PRIMA di decidere se cercare ancora o procedere alla stesura del post.
Passa nel campo 'reflection' la tua analisi: cosa hai trovato, cosa manca, se basta per
scrivere un buon articolo. Non esegue ricerche: serve solo a ragionare in modo esplicito
e tracciabile. Usalo tipicamente DOPO una ricerca web e PRIMA di concludere."""
