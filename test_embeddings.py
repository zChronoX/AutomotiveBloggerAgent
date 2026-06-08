"""
Confronto tra il modello di embedding attuale (all-MiniLM-L6-v2) e il candidato
multilingual-e5-small, per decidere se vale la pena cambiarlo.

Risponde a tre domande:
  1. SONO RETROCOMPATIBILI? (stessa dimensione del vettore? -> NO = serve re-ingest di ChromaDB)
  2. QUANTO CAMBIANO LE SOGLIE? (le similarita' su coppie simili/diverse cambiano -> ritarare 0.75/1.10/1.20)
  3. E5 HA BISOGNO DEI PREFISSI? (query:/passage:) -> confronto con e senza

Uso (serve sentence-transformers, gia' usato dal progetto):
    pip install sentence-transformers --break-system-packages   # se non presente
    python test_embeddings.py

NOTA: la prima esecuzione SCARICA i modelli (qualche centinaio di MB), serve connessione.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Coppie di topic realistiche del dominio automotive.
# SIMILI = dovrebbero superare la soglia (stesso argomento, formulazione diversa)
# DIVERSI = NON dovrebbero superarla (argomenti distinti)
coppie_simili = [
    ("honda cbr650r", "honda cbr 650 r"),
    ("alfa romeo giulia quadrifoglio", "giulia quadrifoglio"),
    ("frenata rigenerativa", "recupero energia in frenata"),
    ("batterie allo stato solido", "batterie stato solido"),
]
coppie_diverse = [
    ("honda cbr650r", "honda cb650r"),          # caso noto delicato: modelli DIVERSI
    ("frenata rigenerativa", "cambio doppia frizione"),
    ("batterie allo stato solido", "sistemi adas"),
    ("ducati panigale", "fiat 600 elettrica"),
]

print("=" * 68)
print("CONFRONTO EMBEDDING: all-MiniLM-L6-v2  vs  multilingual-e5-small")
print("=" * 68)

print("\nCaricamento modelli (la prima volta li scarica)...")
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
e5 = SentenceTransformer("intfloat/multilingual-e5-small")

# --- DOMANDA 1: dimensione dei vettori ---
dim_minilm = minilm.get_sentence_embedding_dimension()
dim_e5 = e5.get_sentence_embedding_dimension()
print(f"\n[1] DIMENSIONE VETTORI")
print(f"    MiniLM : {dim_minilm}")
print(f"    e5     : {dim_e5}")
if dim_minilm == dim_e5:
    print("    -> stessa dimensione (ma il re-ingest di ChromaDB serve COMUNQUE:")
    print("       i vettori non sono confrontabili tra modelli diversi).")
else:
    print(f"    -> DIVERSA: ChromaDB DEVE essere re-ingestato col nuovo modello.")

def embed_minilm(txt):
    return minilm.encode(txt, normalize_embeddings=True)

def embed_e5(txt, prefix=True):
    # e5 e' addestrato con prefissi "query:"/"passage:". Per il matching di topic
    # usiamo "query:" per entrambi (e' un confronto simmetrico topic-vs-topic).
    t = f"query: {txt}" if prefix else txt
    return e5.encode(t, normalize_embeddings=True)

# --- DOMANDA 2 & 3: similarita' e soglie ---
def analizza(coppie, etichetta):
    print(f"\n    {etichetta}")
    print(f"    {'coppia':<48} {'MiniLM':>7} {'e5+pfx':>7} {'e5-pfx':>7}")
    print(f"    {'-'*48} {'-'*7} {'-'*7} {'-'*7}")
    agg = {"minilm": [], "e5p": [], "e5n": []}
    for a, b in coppie:
        s_min = cosine(embed_minilm(a), embed_minilm(b))
        s_e5p = cosine(embed_e5(a, True),  embed_e5(b, True))
        s_e5n = cosine(embed_e5(a, False), embed_e5(b, False))
        agg["minilm"].append(s_min); agg["e5p"].append(s_e5p); agg["e5n"].append(s_e5n)
        etich = f"{a[:22]} ~ {b[:22]}"
        print(f"    {etich:<48} {s_min:>7.3f} {s_e5p:>7.3f} {s_e5n:>7.3f}")
    return agg

print(f"\n[2/3] SIMILARITA' COSENO (e5+pfx = con prefisso query:, e5-pfx = senza)")
sim = analizza(coppie_simili, "COPPIE SIMILI (dovrebbero stare SOPRA la soglia):")
div = analizza(coppie_diverse, "COPPIE DIVERSE (dovrebbero stare SOTTO la soglia):")

print("\n" + "=" * 68)
print("LETTURA DEI RISULTATI")
print("=" * 68)
def media(x): return sum(x)/len(x) if x else 0
print(f"Media SIMILI  -> MiniLM {media(sim['minilm']):.3f} | e5+pfx {media(sim['e5p']):.3f} | e5-pfx {media(sim['e5n']):.3f}")
print(f"Media DIVERSI -> MiniLM {media(div['minilm']):.3f} | e5+pfx {media(div['e5p']):.3f} | e5-pfx {media(div['e5n']):.3f}")
print()
print("Come leggere:")
print(" - Un buon embedding ha un GAP ampio tra media SIMILI e media DIVERSI.")
print("   Confronta il gap di MiniLM con quello di e5: chi separa meglio?")
print(" - La soglia attuale (0.75) vale SOLO per MiniLM. Per e5 la nuova soglia")
print("   va scelta IN MEZZO al gap di e5 (es. a meta' tra le due medie).")
print(" - Se 'e5-pfx' separa molto peggio di 'e5+pfx', allora i prefissi query:/passage:")
print("   sono OBBLIGATORI, e vanno aggiunti in semantic.py e nel vectorstore.")
print(" - Attenzione alla coppia 'cbr650r ~ cb650r' (DIVERSI): se un modello li segna")
print("   troppo simili, rischia di fonderli erroneamente nel KG (gia' caso delicato).")
