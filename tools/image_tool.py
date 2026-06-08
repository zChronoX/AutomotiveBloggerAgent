"""
Tool di generazione delle immagini di copertina per i post approvati.
Uso due API diverse, la prima gratis di PollinationsAI funziona sempre
ma produce risultati scarsi. La seconda usa un modello FLUX, che produce 
risultati di altissima qualità, ma usa l'API di Cloudflare.
Nessuna generaazione delle immagini viene fatta in locale.
"""


import os
import time
import base64
import urllib.parse
import requests
from langchain_core.tools import tool
from prompts.tool_prompts import IMAGE_GENERATOR_PROMPT



# cartella delle copertine generate
COVERS_DIR = "generated_covers"

# Credenziali Cloudflare lette dal .env
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

# Modello usato (si può cambiare se si vuole)
CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"


def _save_bytes(filepath: str, data: bytes) -> str:
    with open(filepath, "wb") as f:
        f.write(data)
    return f"Immagine generata e salvata con successo come '{filepath}'."



# API di generazione delle immagini preferito perché ha qualità altissima
def _genera_flux(prompt: str, filepath: str) -> str:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError("Credenziali Cloudflare assenti nel .env "
                           "(CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN).")

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    payload = {"prompt": prompt, "num_steps": 6}

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Cloudflare ha risposto con status {response.status_code}: {response.text[:200]}")

    # La risposta puo' essere JSON (con immagine in base64) o binaria diretta.
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        dati = response.json()
        image_b64 = (dati.get("result") or {}).get("image")
        if not image_b64:
            raise RuntimeError("Cloudflare: JSON ricevuto ma senza campo immagine.")
        return _save_bytes(filepath, base64.b64decode(image_b64))
    # binaria diretta
    return _save_bytes(filepath, response.content)

# API di generazione delle immagini di fallback in caso di fallimento di Cloudflare
# produce risultati di qualità molto bassa (era inizialmente la prima scelta prima di conoscere Flux 1)
def _genera_pollinations(prompt: str, filepath: str) -> str:
    encoded_prompt = urllib.parse.quote(prompt)
    url = (f"https://image.pollinations.ai/prompt/{encoded_prompt}"
           f"?width=1920&height=1080&nologo=true&model=flux")
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Pollinations status {response.status_code}")
    return _save_bytes(filepath, response.content)




# Tool vero e proprio che genera le immagini
# Genera un'immagine con il timestamp corrente (così evito sovrascritture)
# Utilizzo anche una tecnica di prompt engineering, in cui
# dico al modello di evitare a tutti i costi parole, scritte e loghi
# perché tendono sempre ad apparire.
@tool(description=IMAGE_GENERATOR_PROMPT)
def generate_cover_image(prompt: str, filename: str = None) -> str:
    """
    Genera un'immagine di copertina e la salva in 'generated_covers/'.
    Motore principale: Cloudflare FLUX.1-schnell (cloud, qualita' alta).
    Fallback automatico: Pollinations AI, se Cloudflare non e' disponibile.
    """
    try:
        os.makedirs(COVERS_DIR, exist_ok=True)
    except Exception:
        pass


    if not filename:
        filename = f"copertina_post_{int(time.time())}.png"
    filepath = os.path.join(COVERS_DIR, os.path.basename(filename))

    style_suffix = (
        ", photorealistic, highly detailed, professional automotive photography, "
        "sharp focus, cinematic lighting, clean composition. "
        "NO text, no words, no letters, no captions, no logos, no watermark, "
        "no magazine cover layout, no title text."
    )
    enriched_prompt = f"{prompt.strip()}{style_suffix}"

    # Provo con Flux 1
    try:
        return _genera_flux(enriched_prompt, filepath)
    except Exception as e_flux:
        msg_flux = f"[Copertina] Cloudflare FLUX non disponibile ({e_flux}); provo il fallback Pollinations..."
        print(msg_flux)

    # Altrimenti vado di Pollinations
    try:
        return _genera_pollinations(enriched_prompt, filepath)
    except requests.exceptions.Timeout:
        return "Errore: nessun generatore di immagini ha risposto entro il tempo limite (timeout)."
    except Exception as e_poll:
        return f"Errore nella generazione dell'immagine (Cloudflare e Pollinations falliti): {e_poll}"