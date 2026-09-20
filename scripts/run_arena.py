"""Lanceur du serveur Battle Arena Kahn1 vs Gemini Flash (§LinkedIn Reel).

Usage :
    python scripts/run_arena.py
    python scripts/run_arena.py --port 8000 --open-browser
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn


# Configuration sécurisée de l'encodage console pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def open_browser_delayed(url: str, delay: float = 1.0):
    time.sleep(delay)
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Serveur Web Battle Arena Kahn1 vs Gemini Flash")
    parser.add_argument("--host", default="127.0.0.1", help="Hôte d'écoute (défaut: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port d'écoute (défaut: 8000)")
    parser.add_argument("--reload", action="store_true", help="Activer le hot-reload")
    parser.add_argument("--open-browser", action="store_true", default=True, help="Ouvrir automatiquement le navigateur")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/arena"
    print("\n" + "=" * 65)
    print(" [KAHN1 vs GEMINI FLASH] BATTLE ARENA (OpenJEV)")
    print("=" * 65)
    print(f" -> Interface Web : {url}")
    print(" -> Mode Reel     : Basculez en 'Mode Reel (9:16)' pour filmer !")
    print(f" -> API Swagger   : http://{args.host}:{args.port}/docs")
    print("=" * 65 + "\n")

    if args.open_browser:
        threading.Thread(target=open_browser_delayed, args=(url, 1.2), daemon=True).start()

    uvicorn.run(
        "sysone.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
