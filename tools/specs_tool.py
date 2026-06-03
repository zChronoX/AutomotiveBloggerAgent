"""
Tool per il recupero di specifiche tecniche e storia di veicoli.

Strategia a cascata con due fonti:
  1. API Ninjas (api-ninjas.com): dati tecnici strutturati per auto e moto.
     Richiede NINJAS_API_KEY nel .env. Se manca, viene saltata silenziosamente.
  2. Wikipedia IT: storia, contesto e informazioni enciclopediche.
     Usata come complemento o fallback se API Ninjas non trova risultati.
"""

import os
import re
import requests
import wikipedia
from langchain_core.tools import tool
from prompts.tool_prompts import VEHICLE_SPECS_PROMPT

wikipedia.set_lang("it")


def _fetch_ninjas_api(model_name: str) -> str:
    """
    Recupera dati strutturati da API Ninjas (auto e moto),
    restituendo tutti i campi per il modello piu' recente.
    Implementa retry intelligenti con combinazioni diverse di parametri.
    """
    api_key = os.environ.get("NINJAS_API_KEY")
    if not api_key:
        return ""

    headers = {"X-Api-Key": api_key}

    # Prepariamo le combinazioni di parametri possibili (dal piu' preciso al meno).
    # Gestiamo i brand a UNA parola (es. "Benelli TRK502X") e a DUE parole
    # (es. "Alfa Romeo Giulia", "Aston Martin DB11", "Land Rover Defender").
    words = model_name.split()
    params_list = []

    if len(words) >= 3:
        # Brand a due parole: make = prime 2 parole, model = il resto
        params_list.append({"make": " ".join(words[:2]), "model": " ".join(words[2:])})
    if len(words) >= 2:
        # Brand a una parola: make = prima parola, model = il resto
        params_list.append({"make": words[0], "model": " ".join(words[1:])})

    # Variazione con spazio tra lettere e cifre nel codice modello (es. "TRK502X" -> "TRK 502X")
    if len(words) >= 2:
        model_part = " ".join(words[1:])
        split_model = re.sub(r'([A-Za-z])(\d)', r'\1 \2', model_part)
        if split_model != model_part:
            params_list.append({"make": words[0], "model": split_model})

    # Fallback: nome intero come model, e ultima parola sola
    params_list.append({"model": model_name})
    if len(words) >= 2:
        params_list.append({"model": words[-1]})

    # Deduplica mantenendo l'ordine (combinazioni piu' precise per prime)
    seen = set()
    unique_params = []
    for p in params_list:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            unique_params.append(p)
    params_list = unique_params

    endpoints = [
        ("https://api.api-ninjas.com/v1/cars", "AUTO"),
        ("https://api.api-ninjas.com/v1/motorcycles", "MOTO"),
    ]

    for params in params_list:
        for url, label in endpoints:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code != 200:
                    continue
                data = response.json()
                if not data:
                    continue

                # Ordina per anno di produzione (dal piu' recente)
                def _get_year(item):
                    try:
                        return int(item.get("year", 0))
                    except (TypeError, ValueError):
                        return 0

                data.sort(key=_get_year, reverse=True)
                vehicle = data[0]

                make = vehicle.get("make", "").title()
                model = vehicle.get("model", "").title()
                year = vehicle.get("year", "N/D")

                lines = [
                    f"=== SPECIFICHE TECNICHE ({label}) ===",
                    f"- {make} {model} ({year}):",
                ]
                for key, value in vehicle.items():
                    if key not in ("make", "model", "year"):
                        formatted_key = str(key).replace("_", " ").capitalize()
                        lines.append(f"  {formatted_key}: {value}")

                return "\n".join(lines)
            except Exception:
                continue  # Prossimo tentativo

    return ""


def _fetch_wikipedia(model_name: str) -> str:
    """
    Recupera storia e contesto enciclopedico da Wikipedia IT.
    Strategia resiliente: scorre i risultati di ricerca e restituisce il PRIMO
    risultato utile, anche se non c'e' una corrispondenza esatta col nome cercato.
    """
    try:
        search_results = wikipedia.search(model_name)
        if not search_results:
            return ""

        page = None
        # Proviamo fino a 5 risultati (prima erano 3): aumenta le chance di trovare
        # una pagina utile anche quando il match esatto non esiste.
        for title in search_results[:5]:
            for auto_suggest in (False, True):
                try:
                    page = wikipedia.page(title, auto_suggest=auto_suggest)
                    break
                except wikipedia.exceptions.DisambiguationError as e:
                    # Proviamo le prime opzioni della disambiguazione, non solo la prima
                    for opt in (e.options[:3] if e.options else []):
                        try:
                            page = wikipedia.page(opt, auto_suggest=False)
                            break
                        except Exception:
                            continue
                    if page:
                        break
                except (wikipedia.exceptions.PageError, Exception):
                    continue
            if page:
                break

        if not page:
            return ""

        summary = page.summary
        sentences = summary.split(". ")
        short_summary = ". ".join(sentences[:5]).strip()
        if not short_summary.endswith("."):
            short_summary += "."

        # Segnaliamo se la pagina trovata e' un "best effort" (titolo diverso dalla query),
        # utile a chi legge la trace per capire che non e' una corrispondenza esatta.
        note = ""
        if model_name.lower() not in page.title.lower():
            note = " (risultato piu' pertinente trovato, non corrispondenza esatta)"

        return (
            f"=== STORIA E CONTESTO (Wikipedia) ==={note}\n"
            f"Titolo: {page.title}\n{short_summary}\nFonte: {page.url}"
        )
    except Exception:
        return ""


@tool(description=VEHICLE_SPECS_PROMPT)
def fetch_vehicle_specs(car_model: str) -> str:
    """Recupera specifiche tecniche (API Ninjas) e storia (Wikipedia IT) di un veicolo."""
    sections = []

    # 1. Dati tecnici strutturati (API Ninjas)
    tech_data = _fetch_ninjas_api(car_model)
    if tech_data:
        sections.append(tech_data)

    # 2. Storia e contesto (Wikipedia)
    wiki_data = _fetch_wikipedia(car_model)
    if wiki_data:
        sections.append(wiki_data)

    if not sections:
        return (
            f"Nessuna informazione tecnica ne' pagina Wikipedia trovata per '{car_model}'. "
            f"Prova con un nome piu' specifico o usa 'mcp_web_search' come alternativa."
        )

    return "\n\n".join(sections)
