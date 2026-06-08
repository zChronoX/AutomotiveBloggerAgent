"""
Test diagnostico dei feed RSS automotive.
Prova ogni URL e mostra: se risponde, quante notizie torna, e i primi titoli.
Cosi' decidiamo quali feed tenere nel trend_tool.

Uso (con feedparser installato):
    python test_rss_feeds.py
"""
import feedparser

# Candidati da testare. Aggiungi/togli URL liberamente.
FEEDS = {
    "Motor1":   "https://it.motor1.com/rss/news/all/",
    "Alvolante":"https://www.alvolante.it/rss/news",
    "Motorbox": "https://www.motorbox.com/auto/rss",
    # candidati extra da provare (alcuni potrebbero non esistere/non essere validi):
    "Quattroruote": "https://www.quattroruote.it/rss",
    "Automoto":     "https://www.automoto.it/rss/news.xml",
    "HDmotori":     "https://www.hdmotori.it/feed/",
}

print("=" * 65)
print("TEST FEED RSS AUTOMOTIVE")
print("=" * 65)

risultati_ok = []

for nome, url in FEEDS.items():
    print(f"\n[{nome}] {url}")
    try:
        feed = feedparser.parse(url)
        bozo = getattr(feed, "bozo", 0)
        n = len(feed.entries)

        if bozo and not feed.entries:
            motivo = getattr(feed, "bozo_exception", "feed non valido/irraggiungibile")
            print(f"   FALLITO: {motivo}")
            continue

        if n == 0:
            print("   Risponde ma 0 notizie (feed vuoto?).")
            continue

        # bozo puo' essere 1 anche se il feed e' utilizzabile (warning non fatali)
        avviso = " (con warning non fatali)" if bozo else ""
        print(f"   OK: {n} notizie trovate{avviso}. Primi 5 titoli:")
        for i, entry in enumerate(feed.entries[:5], 1):
            titolo = getattr(entry, "title", "(senza titolo)").strip()
            print(f"      {i}. {titolo}")
        risultati_ok.append((nome, url, n))

    except Exception as e:
        print(f"   ERRORE: {type(e).__name__}: {e}")

print("\n" + "=" * 65)
print("RIEPILOGO — feed utilizzabili:")
if risultati_ok:
    for nome, url, n in risultati_ok:
        print(f"  - {nome}: {n} notizie  ({url})")
    print(f"\nTotale feed validi: {len(risultati_ok)} su {len(FEEDS)}")
    print("Tieni nel trend_tool quelli con piu' notizie e titoli pertinenti.")
else:
    print("  Nessun feed valido. Controlla la connessione o gli URL.")
