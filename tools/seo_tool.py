"""
Tool usato insieme alla generazione delle immagini di coperttina per valutare la bozza finale.
Usiamo l'indice Gulpease (linguistico italiano) anzichè Flesch per maggiore accuratezza (il secondo è per l'inglese).
E' un'indice basato sulla lunghezza delle frasi e sulla lunghezza delle parole.

"""

import re
import textstat
from langchain_core.tools import tool
from prompts.tool_prompts import SEO_ANALYSIS_PROMPT

# Lingua italiana per textstat
textstat.set_lang("it")

# Analizza la densità della keyword e la leggibilità (indice Gulpease, adatto all'italiano).
# Se Gulpease = 100, significa che il testo è estremamente semplice (livello scolastico elementare).
# Se Gulpease = 0, significa che il testo è estremamente complesso (livello universitario avanzato).
# Per un blog tecnico automotive, 40-60 e' il range IDEALE: abbastanza accessibile
# per il pubblico appassionato, senza banalizzare il contenuto tecnico.
@tool(description=SEO_ANALYSIS_PROMPT)
def analyze_seo_and_readability(text: str, target_keyword: str) -> str:
    total_words = len(text.split())

    # Densita' keyword con match a PAROLA INTERA: evita falsi positivi come
    # contare "auto" dentro "automobile" o "autostrada".
    pattern = r"\b" + re.escape(target_keyword.lower()) + r"\b"
    keyword_count = len(re.findall(pattern, text.lower()))
    density = (keyword_count / total_words) * 100 if total_words > 0 else 0.0


    gulpease = textstat.gulpease_index(text)
    gulpease_display = round(max(0.0, min(100.0, gulpease)), 1) 

    # Soglie standard Gulpease (riferite al livello di istruzione del lettore)
    # Per un blog tecnico automotive, 40-60 e' il range IDEALE: abbastanza accessibile
    # per il pubblico appassionato, senza banalizzare il contenuto tecnico.
    if gulpease >= 80:
        read_level = "Molto facile (testo elementare), troppo semplice per un blog tecnico"
    elif gulpease >= 60:
        read_level = "Facile (leggibile da tutti), buono per articoli divulgativi"
    elif gulpease >= 40:
        read_level = "Adeguato per un blog tecnico, accessibile a lettori appassionati"
    else:
        read_level = "Molto tecnico (potrebbe escludere i lettori meno esperti)"

    # Giudizio sulla densita' keyword (range SEO standard: 0.5% - 2.5%)
    if density < 0.3:
        kw_level = "troppo bassa, la keyword e' poco presente, considera di usarla di piu'"
    elif density < 0.5:
        kw_level = "leggermente bassa, aumentare un po' migliorerebbe il posizionamento"
    elif density <= 2.5:
        kw_level = "nel range ottimale SEO"
    else:
        kw_level = "troppo alta, c'è rischio di keyword stuffing, riducila un po'"

    return (
        "Analisi SEO completata:\n"
        f"Parola chiave '{target_keyword}' trovata {keyword_count} volte "
        f"Densità: {density:.2f}% su {total_words} parole - {kw_level}.\n"
        f"Leggibilità (Gulpease): {gulpease_display}/100 - {read_level}."
    )