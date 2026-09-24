Installe Kahn1 sur cette machine et prouve qu'il fonctionne avec une vraie requête.

Kahn1 (https://github.com/Okura66/kahn1, MIT) est un moteur de décision open source : un Qwen2.5-3B affiné qui répond à des questions typées (choice, score, noul) sur un texte en lisant les probabilités des tokens d'option. Il est servi par une application FastAPI, `sysone`. Guide : https://kahn1.com/fr/demarrer/

Règles : avance étape par étape et montre chaque commande avant de la lancer. Demande-moi avant d'utiliser sudo, d'installer des paquets système ou de modifier quoi que ce soit hors du dossier du projet. Si une étape échoue, montre l'erreur exacte, explique la cause probable et propose une correction avant de continuer. N'invente jamais de sortie : si le modèle n'a pas pu tourner, dis-le.

1. Inspecte la machine et dis-moi ce que tu trouves : système (Linux, macOS, Windows, WSL2), version de Python (3.11+ requise), RAM et disque libres, et si un GPU NVIDIA est utilisable (`nvidia-smi`). Choisis ensuite le backend et explique pourquoi :
   - GPU (vLLM) : seulement sous Linux ou WSL2 avec un GPU NVIDIA et CUDA, 8 Go de VRAM ou plus.
   - CPU (transformers) sinon : environ 13 Go de RAM libre, quelques secondes par question.
   Le téléchargement du modèle fait environ 6 Go dans les deux cas.

2. Installe uv s'il manque (https://docs.astral.sh/uv/). Si ce dossier n'est pas déjà un clone du dépôt, clone-le :
   git clone https://github.com/Okura66/kahn1 && cd kahn1
   uv venv --python 3.11
   GPU : uv pip install -e ".[gpu]"
   CPU : uv pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
   Le tokenizer du modèle demande transformers 5 ou plus. Vérifie avec
   uv run python -c "import transformers; print(transformers.__version__)"
   et lance `uv pip install -U transformers` s'il affiche 4.x.

3. Démarre le serveur en arrière-plan et garde son journal :
   GPU : SYSONE_MODEL=Okura66/Kahn1-Qwen2.5-3B uv run sysone serve --port 8000
   CPU : SYSONE_BACKEND=cpu SYSONE_MODEL=Okura66/Kahn1-Qwen2.5-3B uv run sysone serve --port 8000
   Sous Windows PowerShell, définis d'abord chaque variable avec $env:NOM = "valeur". Interroge GET http://127.0.0.1:8000/health jusqu'à obtenir {"status": "ok"}.

4. Envoie ce JSON en POST à http://127.0.0.1:8000/v1/evaluate/jev (écris-le dans un fichier et envoie-le avec curl -d @fichier, ou utilise Python). Le premier appel charge le modèle et peut prendre des minutes :
   {"state": "Bonjour, j'ai été débité deux fois pour ma commande #48213. Je veux être remboursé, sinon je fais opposition auprès de ma banque.",
    "schema": {
      "intent": {"type": "choice", "instructions": "Quelle est la demande principale du client ?",
                 "criteria": {"remboursement": "le client veut être remboursé", "livraison": "question sur une livraison", "compte": "problème d'accès au compte"}},
      "urgency": {"type": "score", "instructions": "Quel est le niveau d'urgence de ce message ?", "criteria": ["faible", "moyen", "élevé", "critique"]},
      "churn_risk": {"type": "noul", "instructions": "Le client menace de quitter le service ou d'escalader."}},
    "n_permutations": 1}
   Montre-moi la réponse JSON et vérifie-la : intent a un choice et des probabilités dont la somme vaut 1, urgency a un level et un score, churn_risk est un nombre entre 0 et 1. Donne latency_ms, puis renvoie la même requête pour montrer la latence à chaud.

5. Termine par un court résumé : backend utilisé, chemin d'installation, comment arrêter et relancer le serveur, et où aller ensuite (https://kahn1.com/fr/demarrer/ pour l'API, le format de réponse, les réglages et la calibration sur mes propres données).
