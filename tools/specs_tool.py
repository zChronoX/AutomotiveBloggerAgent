"""
Tool per il recupero di specifiche tecniche e storia di veicoli.
Viene chiamato quando è necessario recuperare dati specifici su un veicolo,
come nel caso di una recensione. Non dovrebbe essere chiamato a due volte
per un confronto tra veicoli dato che c'è tool apposito.
Il tool cerca prima sul sito di API-Ninjas e poi su Wikipedia.
Spesso Wikipedia non ha i modelli esatti, pertanto uso una 
seconda API, che teoricamente dovrebbe tornare i dati esatti di un modello.
"""

import os
import re
import requests
import wikipedia
from langchain_core.tools import tool
from prompts.tool_prompts import VEHICLE_SPECS_PROMPT

wikipedia.set_lang("it")

# Parte del tool che cerca su API Ninjas
# è stato necessario implementare un controllo per i brand a due nomi
# come "Alfa Romeo" o "Aston Martin", ecc. Sennò capitava che il tool cercasse
# per "Romeo" o "Martin" (come modello).
# l'API usate sono 2, una per le auto e una per le moto, entrambe gratuite.
def _fetch_ninjas_api(model_name: str) -> str:
    api_key = os.environ.get("NINJAS_API_KEY")
    if not api_key:
        return ""

    headers = {"X-Api-Key": api_key}

    words = model_name.split()
    params_list = []

    if len(words) >= 3:
        params_list.append({"make": " ".join(words[:2]), "model": " ".join(words[2:])})
    if len(words) >= 2:
        params_list.append({"make": words[0], "model": " ".join(words[1:])})

    # E' possibile che alcuni modelli abbiano uno spazio 
    if len(words) >= 2:
        model_part = " ".join(words[1:])
        split_model = re.sub(r'([A-Za-z])(\d)', r'\1 \2', model_part)
        if split_model != model_part:
            params_list.append({"make": words[0], "model": split_model})

    params_list.append({"model": model_name})
    if len(words) >= 2:
        params_list.append({"model": words[-1]})


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

                # Ordina per anno di produzione 
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
                    f"  <Specifiche tecniche estratte ({label})> ",
                    f"- {make} {model} ({year}):",
                ]
                for key, value in vehicle.items():
                    if key not in ("make", "model", "year"):
                        formatted_key = str(key).replace("_", " ").capitalize()
                        lines.append(f"  {formatted_key}: {value}")

                return "\n".join(lines)
            except Exception:
                continue  

    return ""

# Nel caso in cui l'API di sopra non dovesse tornare nulla, si prova a cercare
# su Wikipedia in italiano. 
def _fetch_wikipedia(model_name: str) -> str:
    try:
        search_results = wikipedia.search(model_name)
        if not search_results:
            return ""

        page = None
        # Cerco tra i primi 7 risultati (prima erano 5) in modo da aumentare le chance di trovare
        # una pagina utile anche quando il match esatto non esiste.
        for title in search_results[:7]:
            for auto_suggest in (False, True):
                try:
                    page = wikipedia.page(title, auto_suggest=auto_suggest)
                    break
                except wikipedia.exceptions.DisambiguationError as e:
                    # Proviamo le prime opzioni della disambiguazione, non solo la prima
                    for opt in (e.options[:4] if e.options else []):
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

        # Segnalo che la pagina trovata non e' una corrispondenza esatta.
        note = ""
        if model_name.lower() not in page.title.lower():
            note = " (risultato piu' pertinente trovato, non corrispondenza esatta)"

        return (
            f"  Storia estratta da Wikipedia {note}\n"
            f"  Titolo pagina: {page.title}\n{short_summary}\nFonte: {page.url}"
        )
    except Exception:
        return ""

# Tool che recupera i dati dalle API Ninjas e da Wikipedia.
# Poi li restituisce in un formato strutturato, in modo che l'agente possa usarli.
@tool(description=VEHICLE_SPECS_PROMPT)
def fetch_vehicle_specs(car_model: str) -> str:
    sections = []

    # Dati tecnici strutturati
    tech_data = _fetch_ninjas_api(car_model)
    if tech_data:
        sections.append(tech_data)

    # Storia e contesto
    wiki_data = _fetch_wikipedia(car_model)
    if wiki_data:
        sections.append(wiki_data)

    if not sections:
        return (
            f"Nessuna informazione tecnica ne' pagina Wikipedia trovata per '{car_model}'. "
            f"Prova con un nome piu' specifico o usa 'mcp_web_search' come alternativa."
        )

    return "\n\n".join(sections)
