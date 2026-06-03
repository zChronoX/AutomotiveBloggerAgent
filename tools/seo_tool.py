import re
import textstat
from langchain_core.tools import tool
from prompts.tool_prompts import SEO_ANALYSIS_PROMPT

# Lingua italiana per textstat (rilevante per le metriche linguistiche)
textstat.set_lang("it")


@tool(description=SEO_ANALYSIS_PROMPT)
def analyze_seo_and_readability(text: str, target_keyword: str) -> str:
    """Analizza densita' della keyword e leggibilita' (indice Gulpease, adatto all'italiano)."""
    total_words = len(text.split())

    # Densita' keyword con match a PAROLA INTERA: evita falsi positivi come
    # contare "auto" dentro "automobile" o "autostrada".
    pattern = r"\b" + re.escape(target_keyword.lower()) + r"\b"
    keyword_count = len(re.findall(pattern, text.lower()))
    density = (keyword_count / total_words) * 100 if total_words > 0 else 0.0

    # Leggibilita': indice GULPEASE, progettato per l'italiano (a differenza di Flesch,
    # tarato sull'inglese). Scala teorica 0-100: piu' alto = piu' leggibile.
    gulpease = textstat.gulpease_index(text)
    gulpease_display = round(max(0.0, min(100.0, gulpease)), 1)  # clamp per testi limite

    # Soglie standard Gulpease (riferite al livello di istruzione del lettore)
    # Per un blog tecnico automotive, 40-60 e' il range IDEALE: abbastanza accessibile
    # per il pubblico appassionato, senza banalizzare il contenuto tecnico.
    if gulpease >= 80:
        read_level = "Molto facile (testo elementare — troppo semplice per un blog tecnico)"
    elif gulpease >= 60:
        read_level = "Facile (leggibile da tutti — buono per articoli divulgativi)"
    elif gulpease >= 40:
        read_level = "Adeguato per un blog tecnico (accessibile a lettori appassionati)"
    else:
        read_level = "Molto tecnico (potrebbe escludere i lettori meno esperti)"

    # Giudizio sulla densita' keyword (range SEO standard: 0.5% - 2.5%)
    if density < 0.3:
        kw_level = "troppo bassa — la keyword e' poco presente, considera di usarla di piu'"
    elif density < 0.5:
        kw_level = "leggermente bassa — aumentare un po' migliorerebbe il posizionamento"
    elif density <= 2.5:
        kw_level = "nel range ottimale SEO (0.5%-2.5%)"
    else:
        kw_level = "troppo alta — rischio keyword stuffing, riducila un po'"

    return (
        "Analisi SEO completata:\n"
        f"- Parola chiave '{target_keyword}' trovata {keyword_count} volte "
        f"(densita': {density:.2f}% su {total_words} parole) — {kw_level}.\n"
        f"- Leggibilita' (Gulpease): {gulpease_display}/100 — {read_level}."
    )