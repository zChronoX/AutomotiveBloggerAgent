import os
import time
import base64
import urllib.parse
import requests
from langchain_core.tools import tool
from prompts.tool_prompts import IMAGE_GENERATOR_PROMPT

# Cartella dedicata alle copertine generate (creata se non esiste):
# evita di affollare la cartella principale del progetto.
COVERS_DIR = "generated_covers"

# --- Credenziali Cloudflare (lette dal .env, MAI hardcoded) ---
# Nel .env servono: CLOUDFLARE_ACCOUNT_ID e CLOUDFLARE_API_TOKEN
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"


def _save_bytes(filepath: str, data: bytes) -> str:
    with open(filepath, "wb") as f:
        f.write(data)
    return f"Immagine generata e salvata con successo come '{filepath}'."


def _genera_flux(prompt: str, filepath: str) -> str:
    """
    Motore PRINCIPALE: Cloudflare Workers AI con FLUX.1-schnell.
    Servizio cloud (non usa la GPU locale). Restituisce una stringa di esito;
    solleva un'eccezione se fallisce, cosi' il chiamante puo' attivare il fallback.
    """
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError("Credenziali Cloudflare assenti nel .env "
                           "(CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN).")

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    payload = {"prompt": prompt, "num_steps": 4}

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


def _genera_pollinations(prompt: str, filepath: str) -> str:
    """
    Motore di FALLBACK: Pollinations AI (Flux gratuito).
    Usato solo se Cloudflare non e' disponibile/fallisce.
    """
    encoded_prompt = urllib.parse.quote(prompt)
    url = (f"https://image.pollinations.ai/prompt/{encoded_prompt}"
           f"?width=1920&height=1080&nologo=true&model=flux")
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Pollinations status {response.status_code}")
    return _save_bytes(filepath, response.content)


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

    # Nome file UNICO di default: evita di sovrascrivere copertine precedenti.
    if not filename:
        filename = f"copertina_post_{int(time.time())}.png"
    filepath = os.path.join(COVERS_DIR, os.path.basename(filename))

    # Arricchiamo il prompt con direttive di STILE e direttive NEGATIVE: i modelli di
    # generazione (FLUX in particolare) tendono ad aggiungere finte scritte/loghi/layout da
    # rivista quando il prompt evoca "copertina". Forziamo fotorealismo sul soggetto e
    # vietiamo qualsiasi testo. Questo suffisso vale sia per FLUX che per Pollinations.
    style_suffix = (
        ", photorealistic, highly detailed, professional automotive photography, "
        "sharp focus, cinematic lighting, clean composition. "
        "NO text, no words, no letters, no captions, no logos, no watermark, "
        "no magazine cover layout, no title text."
    )
    enriched_prompt = f"{prompt.strip()}{style_suffix}"

    # 1) Tentativo con Cloudflare FLUX (principale)
    try:
        return _genera_flux(enriched_prompt, filepath)
    except Exception as e_flux:
        msg_flux = f"[Copertina] Cloudflare FLUX non disponibile ({e_flux}); provo il fallback Pollinations..."
        print(msg_flux)

    # 2) Fallback: Pollinations
    try:
        return _genera_pollinations(enriched_prompt, filepath)
    except requests.exceptions.Timeout:
        return "Errore: nessun generatore di immagini ha risposto entro il tempo limite (timeout)."
    except Exception as e_poll:
        return f"Errore nella generazione dell'immagine (Cloudflare e Pollinations falliti): {e_poll}"