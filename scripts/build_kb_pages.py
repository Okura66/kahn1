"""Build the knowledge-base pages of kahn1.com (EN + FR) on the redesign's shell.

Each page is cut from the caveats page (EN) or the vigilance page (FR): the shared markup
(consent, analytics, fonts, sidebar, footer) stays byte for byte, only the head metadata, the
JSON-LD and <main> change. Also writes docs/data/landscape.csv and adds the new pages to the
sidebars, the sitemap and llms.txt.

    python scripts/build_kb_pages.py
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "docs"
SITE = "https://kahn1.com"
TODAY = "2026-10-01"   # the landscape was checked on this date
MODIFIED = "2026-10-02"
N = "&nbsp;"

# ---------------------------------------------------------------------------------------------
# Landscape data (checked on the providers' pages, 2026-10-01). One row per product or model.
# Vendor claims (calibration, speed) are reported as claims; an empty cell = not on the page.
# ---------------------------------------------------------------------------------------------
FAMILIES = {
    "hosted": ("Hosted decision models", "Modèles de décision hébergés"),
    "open": ("Open decision models", "Modèles de décision ouverts"),
    "llm": ("General LLMs with structured outputs", "LLM généralistes avec structured outputs"),
    "decoding": ("Constrained decoding and validation", "Décodage contraint et validation"),
    "cloud": ("Cloud classification APIs", "API de classification cloud"),
    "encoder": ("Zero-shot and few-shot encoders", "Encodeurs zero-shot et few-shot"),
    "guard": ("Guard and judge models", "Modèles guardrails et juges"),
}
# name, family, provider, licence, deployment, output, probabilities, price, url[, url of the probabilities cell]
ROWS = [
    ("Jev 1.13.0", "hosted", "TypeSafe", "Closed", "TypeSafe API, OpenRouter", "Label + probabilities (choice, score, noul)", "Calibrated according to the vendor", "$0.042 per million input tokens, output free", "https://docs.typesafe.ai/models"),
    ("d1", "hosted", "Liquid AI", "Closed", "Liquid API", "Label + probabilities", "Calibrated according to the vendor", "Free tier (d1:free); paid rates not published", "https://docs.liquid.ai/lfm/models/decision-models"),
    ("Decisions API", "hosted", "OpenAI", "Closed", "API, limited preview", "", "", "", "https://openai.com/index/devday-2026-recap"),
    ("Clef, Clef-flash", "open", "Cloudflare", "Apache 2.0", "Workers AI, weights on Hugging Face; 27B and 9B", "Label + probabilities, Jev-API compatible", "Calibrated according to the vendor", "Open weights; hosted on Workers AI", "https://blog.cloudflare.com/clef-decision-models/"),
    ("JevK5 v0.3", "open", "allebee", "Apache 2.0 (weights and code)", "Local GPU or CPU (GGUF), /v1/systemone request shape; 4B and 9B", "Label + probabilities", "ECE published by the authors", "Free", "https://github.com/allebee/jevk5"),
    ("Kahn1 4B, Kahn1 3B", "open", "Kahn1", "4B Apache 2.0, 3B Qwen Research License, code MIT", "Local GPU or CPU, browser (3B); Jev question fields at /v1/evaluate/jev", "Label + probabilities", "Temperature-calibrated, ECE published", "Free", "https://kahn1.com/models/"),
    ("Tev1-4B-experimental", "open", "Together AI", "Being finalized (base Qwen3.5-4B, Apache 2.0)", "Together API, weights on Hugging Face", "Label (one option letter)", "Calibration not comprehensively evaluated (model card)", "", "https://huggingface.co/togethercomputer/Tev1-4B-experimental"),
    ("open-alternative-jev (so1)", "open", "ikermoel", "Apache 2.0", "Any open LLM through Hugging Face or vLLM", "Label + probabilities", "Temperature scaling, ECE published", "Free", "https://github.com/ikermoel/open-alternative-jev"),
    ("Laya", "open", "Convai Innovations", "Apache 2.0", "ModernBERT 421M or mmBERT 322M; laya-serve exposes /v1/systemone", "Label + probabilities", "Calibrated according to the vendor, ECE published", "Free", "https://huggingface.co/convaiinnovations/laya"),
    ("Structured Outputs", "llm", "OpenAI", "Closed", "API", "Generated JSON", "Logprobs, not when reasoning is on", "Per input and output token", "https://developers.openai.com/api/docs/guides/structured-outputs", "https://developers.openai.com/api/docs/guides/latest-model"),
    ("Structured outputs", "llm", "Anthropic", "Closed", "API", "Generated JSON", "No logprobs documented", "Per input and output token", "https://platform.claude.com/docs/en/build-with-claude/structured-outputs"),
    ("Structured outputs", "llm", "Google", "Closed", "Gemini API, Vertex AI", "Generated JSON or enum", "Logprobs on Vertex AI", "Per input and output token", "https://ai.google.dev/gemini-api/docs/structured-output", "https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/"),
    ("Moderation", "guard", "OpenAI", "Closed", "API", "Flags + scores, 13 categories", "Scores 0 to 1, to recalibrate when the model changes", "Free", "https://developers.openai.com/api/docs/guides/moderation"),
    ("Outlines", "decoding", "dottxt", "Apache 2.0", "Local, any model", "Choice, regex, JSON", "Not documented per option", "Free", "https://github.com/dottxt-ai/outlines"),
    ("XGrammar", "decoding", "MLC", "Apache 2.0", "Inside vLLM, SGLang, TensorRT-LLM", "JSON, grammars", "Not documented per option", "Free", "https://github.com/mlc-ai/xgrammar"),
    ("vLLM structured outputs", "decoding", "vLLM", "", "Local", "Choice, regex, JSON, grammar", "Not documented per option", "Free", "https://docs.vllm.ai/en/latest/features/structured_outputs.html"),
    ("GBNF grammars", "decoding", "llama.cpp", "", "Local", "JSON, grammars", "Not documented per option", "Free", "https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md"),
    ("Guidance", "decoding", "guidance-ai (from Microsoft)", "MIT", "Local", "select, regex, JSON", "Not documented per option", "Free", "https://github.com/guidance-ai/guidance"),
    ("Instructor", "decoding", "567 Labs", "MIT", "Over provider APIs", "Validated JSON, retries", "", "Free", "https://github.com/567-labs/instructor"),
    ("BAML", "decoding", "Boundary", "Apache 2.0", "Over provider APIs", "Typed functions", "", "Free", "https://github.com/BoundaryML/baml"),
    ("Cloud Natural Language classifyText", "cloud", "Google", "Closed", "API", "Category + confidence, fixed taxonomy", "Confidence per category", "", "https://docs.cloud.google.com/natural-language/docs/classifying-text"),
    ("Comprehend custom classification", "cloud", "AWS", "Closed", "API, real time or batch", "Label + confidence", "Confidence", "", "https://docs.aws.amazon.com/comprehend/latest/dg/how-document-classification.html"),
    ("Custom text classification", "cloud", "Microsoft Azure", "Closed, retires 2029-03-31", "API", "Label + confidence", "Confidence", "", "https://learn.microsoft.com/en-us/azure/ai-services/language-service/custom-text-classification/overview"),
    ("AI Content Safety", "guard", "Microsoft Azure", "Closed", "API", "Severity levels", "", "F0 and S0 tiers", "https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview"),
    ("Classify", "cloud", "Cohere", "Deprecated (September 2025)", "", "Label + confidences", "", "", "https://docs.cohere.com/docs/deprecations"),
    ("bart-large-mnli", "encoder", "Meta", "MIT", "CPU, 0.4B, NLI", "Label + probabilities", "No calibration claim", "Free", "https://huggingface.co/facebook/bart-large-mnli"),
    ("deberta-v3-large-zeroshot-v2.0", "encoder", "Moritz Laurer", "MIT", "CPU, 0.4B, NLI", "Label + probabilities", "No calibration claim", "Free", "https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"),
    ("ModernBERT-large-zeroshot-v2.0", "encoder", "Moritz Laurer", "Apache 2.0", "CPU, 0.4B, NLI", "Label + probabilities", "No calibration claim", "Free", "https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0"),
    ("GLiClass", "encoder", "Knowledgator", "Apache 2.0", "CPU, every label in one pass", "Label + probabilities", "No calibration claim", "Free", "https://github.com/Knowledgator/GLiClass"),
    ("SetFit", "encoder", "Hugging Face", "Apache 2.0", "CPU, few-shot fine-tuning", "Label + probabilities", "No calibration claim", "Free", "https://github.com/huggingface/setfit"),
    ("Llama Guard 4", "guard", "Meta", "Llama 4 Community License", "Local, 12B", "Safe or unsafe + categories S1 to S14", "", "Free", "https://huggingface.co/meta-llama/Llama-Guard-4-12B"),
    ("ShieldGemma", "guard", "Google", "Gemma terms", "Local, 2B to 27B", "Policy score", "P(Yes)", "Free", "https://huggingface.co/google/shieldgemma-2b"),
    ("Granite Guardian 4.1", "guard", "IBM", "Apache 2.0", "Local, 8B", "Yes or no, your own criteria", "", "Free", "https://huggingface.co/ibm-granite/granite-guardian-4.1-8b"),
    ("Qwen3Guard", "guard", "Qwen", "Apache 2.0", "Local, 0.6B to 8B", "Safe, unsafe or controversial + categories", "", "Free", "https://github.com/QwenLM/Qwen3Guard"),
    ("gpt-oss-safeguard", "guard", "OpenAI", "Apache 2.0", "Local, 20B and 120B", "Label + reasoning, policy given at inference", "", "Free", "https://huggingface.co/openai/gpt-oss-safeguard-20b"),
    ("Prometheus 2", "guard", "prometheus-eval", "Apache 2.0", "Local, 7B and 8x7B", "Score 1 to 5 + feedback", "", "Free", "https://huggingface.co/prometheus-eval/prometheus-7b-v2.0"),
    ("Flow Judge v0.1", "guard", "Flow AI", "Apache 2.0", "Local, 3.8B", "Score + feedback", "", "Free", "https://huggingface.co/flowaicom/Flow-Judge-v0.1"),
    ("Lynx", "guard", "Patronus AI", "CC BY-NC 4.0 (non-commercial)", "Local, 8B and 70B", "PASS or FAIL + reasoning", "", "Free, non-commercial", "https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct"),
    ("HHEM-2.1-Open", "guard", "Vectara", "Apache 2.0", "Local, 0.1B", "Consistency score", "Score 0 to 1", "Free", "https://huggingface.co/vectara/hallucination_evaluation_model"),
]

FR_CELLS = {
    "Closed": "Fermée", "Free": "Gratuit", "Local": "Local", "API": "API",
    "Label + probabilities": "Label + probabilités", "Generated JSON": "JSON généré",
    "Not documented per option": "Non documentées par option", "No calibration claim": "Aucune calibration annoncée",
    "Per input and output token": "Par token d'entrée et de sortie", "Per call": "À l'appel",
    "Calibrated according to the vendor": "Calibrées selon l'éditeur", "No logprobs": "Pas de logprobs",
}
FR_FREE_TEXT = {
    "$0.042 per million input tokens, output free": "0,042 $ par million de tokens d'entrée, sortie gratuite",
    "Free (d1:free) at launch, pricing to come": "Gratuit (d1:free) au lancement, tarif à venir",
    "Open weights; hosted on Workers AI": "Poids ouverts ; hébergé sur Workers AI",
    "Free, non-commercial": "Gratuit, hors usage commercial",
    "Label + probabilities (choice, score, noul)": "Label + probabilités (choice, score, noul)",
    "Label + probabilities, JEV-API compatible": "Label + probabilités, compatible API JEV",
    "Label (one option letter)": "Label (une lettre d'option)",
    "Logprobs, not on most reasoning models": "Logprobs, pas sur la plupart des modèles de raisonnement",
    "Logprobs on Vertex AI": "Logprobs sur Vertex AI",
    "Generated JSON or enum": "JSON généré ou enum",
    "Flags + scores, 13 categories": "Signalements + scores, 13 catégories",
    "Scores 0 to 1, to recalibrate when the model changes": "Scores de 0 à 1, à recalibrer quand le modèle change",
    "Temperature-calibrated, ECE published": "Calibration par température, ECE publiée",
    "ECE published by the authors": "ECE publiée par les auteurs",
    "Calibration not comprehensively evaluated (model card)": "Calibration non évaluée en détail (fiche du modèle)",
    "Category + confidence, fixed taxonomy": "Catégorie + confiance, taxonomie fixe",
    "Label + confidence": "Label + confiance", "Label + confidences": "Label + confiances",
    "Confidence": "Confiance", "Confidence per category": "Confiance par catégorie",
    "Severity levels": "Niveaux de sévérité", "Policy score": "Score de politique",
    "First-token probability": "Probabilité du premier token",
    "Safe or unsafe + categories S1 to S14": "Sûr ou non + catégories S1 à S14",
    "Yes or no, your own criteria": "Oui ou non, critères libres",
    "Safe, unsafe or controversial + categories": "Sûr, non sûr ou controversé + catégories",
    "Label + reasoning, policy given at inference": "Label + raisonnement, politique donnée à l'inférence",
    "Score 1 to 5 + feedback": "Note de 1 à 5 + commentaire", "Score + feedback": "Note + commentaire",
    "PASS or FAIL + reasoning": "PASS ou FAIL + raisonnement", "Consistency score": "Score de cohérence",
    "Score 0 to 1": "Score de 0 à 1", "Validated JSON, retries": "JSON validé, nouvelles tentatives",
    "Typed functions": "Fonctions typées", "Choice, regex, JSON": "Choix, regex, JSON",
    "JSON, grammars": "JSON, grammaires", "Choice, regex, JSON, grammar": "Choix, regex, JSON, grammaire",
    "select, regex, JSON": "select, regex, JSON",
    "Closed, retires 2029-03-31": "Fermée, retrait le 31 mars 2029",
    "Deprecated (September 2025)": "Retirée (septembre 2025)",
    "Being finalized (base Qwen3.5-4B, Apache 2.0)": "En cours (base Qwen3.5-4B, Apache 2.0)",
    "Apache 2.0 (weights and code)": "Apache 2.0 (poids et code)",
    "4B Apache 2.0, 3B Qwen Research License, code MIT": "4B Apache 2.0, 3B licence de recherche Qwen, code MIT",
    "CC BY-NC 4.0 (non-commercial)": "CC BY-NC 4.0 (non commercial)",
    "TypeSafe API, OpenRouter": "API TypeSafe, OpenRouter", "Liquid API": "API Liquid",
    "Workers AI, weights on Hugging Face; 27B and 9B": "Workers AI, poids sur Hugging Face ; 27B et 9B",
    "Local GPU or CPU (GGUF), /v1/systemone request shape; 4B and 9B": "Local GPU ou CPU (GGUF), requêtes au format /v1/systemone ; 4B et 9B",
    "Local GPU or CPU, browser (3B); accepts the JEV schema format": "Local GPU ou CPU, navigateur (3B) ; accepte le format de schéma JEV",
    "Together serverless, weights on Hugging Face": "Together serverless, poids sur Hugging Face",
    "Any open LLM through Hugging Face or vLLM": "N'importe quel LLM ouvert via Hugging Face ou vLLM",
    "ModernBERT 421M or mmBERT 322M, to fine-tune": "ModernBERT 421M ou mmBERT 322M, à fine-tuner",
    "Gemini API, Vertex AI": "API Gemini, Vertex AI", "Inside vLLM, SGLang, TensorRT-LLM": "Dans vLLM, SGLang, TensorRT-LLM",
    "Local, any model": "Local, n'importe quel modèle", "Local, CPU": "Local, CPU",
    "Over provider APIs": "Sur les API des fournisseurs", "API, real time or batch": "API, temps réel ou par lots",
    "CPU, 0.4B, NLI": "CPU, 0,4B, NLI", "CPU, 151M, every label in one pass": "CPU, 151M, toutes les classes en une passe",
    "CPU, few-shot fine-tuning": "CPU, fine-tuning few-shot", "Local, 12B": "Local, 12B", "Local, 2B to 27B": "Local, 2B à 27B",
    "Local, 8B": "Local, 8B", "Local, 0.6B to 8B": "Local, 0,6B à 8B", "Local, 20B and 120B": "Local, 20B et 120B",
    "Local, 7B and 8x7B": "Local, 7B et 8x7B", "Local, 3.8B": "Local, 3,8B", "Local, 8B and 70B": "Local, 8B et 70B",
    "Local, 0.1B": "Local, 0,1B",
    "Free tier (d1:free); paid rates not published": "Palier gratuit (d1:free) ; tarifs payants non publiés",
    "API, limited preview": "API, accès anticipé limité",
    "Calibrated according to the vendor, ECE published": "Calibrées selon l'éditeur, ECE publiée",
    "Temperature scaling, ECE published": "Calibration par température, ECE publiée",
    "ModernBERT 421M or mmBERT 322M; laya-serve exposes /v1/systemone": "ModernBERT 421M ou mmBERT 322M ; laya-serve expose /v1/systemone",
    "Logprobs, not when reasoning is on": "Logprobs, pas quand le raisonnement est activé",
    "No logprobs documented": "Aucun logprob documenté",
    "guidance-ai (from Microsoft)": "guidance-ai (issu de Microsoft)",
    "Together API, weights on Hugging Face": "API Together, poids sur Hugging Face",
    "CPU, every label in one pass": "CPU, toutes les classes en une passe",
    "Label + probabilities, Jev-API compatible": "Label + probabilités, compatible API Jev",
    "Local GPU or CPU, browser (3B); Jev question fields at /v1/evaluate/jev": "Local GPU ou CPU, navigateur (3B) ; champs de question de Jev sur /v1/evaluate/jev", "F0 and S0 tiers": "Paliers F0 et S0", "Gemma terms": "Conditions Gemma",
}


def fr_cell(s: str) -> str:
    return FR_FREE_TEXT.get(s, FR_CELLS.get(s, s))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def landscape_tables(fr: bool) -> str:
    hdr = (["Produit", "Éditeur", "Licence", "Déploiement", "Ce qui revient", "Probabilités", "Prix"] if fr else
           ["Product", "Provider", "Licence", "Deployment", "Output", "Probabilities", "Price"])
    out = []
    for i, (fam, (en, frn)) in enumerate(FAMILIES.items(), start=2):
        rows = [r for r in ROWS if r[1] == fam]
        if not rows:
            continue
        name = frn if fr else en
        sid = f"f-{fam}"
        out.append(f'    <section class="sec doc" id="{sid}" aria-labelledby="h-{sid}">\n'
                   f'      <h2 id="h-{sid}">{i:02d} · {name}</h2>\n'
                   f'      <div class="tbl-wrap"><table class="tbl"><thead><tr>'
                   + "".join(f'<th scope="col">{h}</th>' for h in hdr) + "</tr></thead><tbody>")
        for name_, _, prov, lic, dep, outp, prob, price, url, *prob_url in rows:
            c = (lambda s: esc(fr_cell(s))) if fr else esc
            pc = f'<a href="{prob_url[0]}" rel="noopener">{c(prob)}</a>' if prob_url and prob else c(prob)
            if fr and url == "https://kahn1.com/models/":
                url = "/fr/modeles/"
            out.append(f'<tr><td><a href="{url}" rel="noopener">{esc(name_)}</a></td><td>{c(prov)}</td><td>{c(lic)}</td>'
                       f'<td>{c(dep)}</td><td>{c(outp)}</td><td>{pc}</td><td>{c(price)}</td></tr>')
        out.append("</tbody></table></div>\n    </section>\n")
    return "\n".join(out)


def write_csv():
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["name", "family", "provider", "licence", "deployment", "output", "probabilities", "price", "source",
                "probabilities_source", "checked"])
    for r in ROWS:
        w.writerow([r[0], FAMILIES[r[1]][0], *r[2:9], r[9] if len(r) > 9 else "", TODAY])
    p = ROOT / "data" / "landscape.csv"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(buf.getvalue().encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# Page bodies
# ---------------------------------------------------------------------------------------------
def toc(items: list[tuple[str, str]], fr: bool) -> str:
    lab = "Sur cette page" if fr else "On this page"
    lis = "".join(f'\n          <li><a href="#{i}">{t}</a></li>' for i, t in items)
    return f'      <nav class="toc" aria-label="{lab}">\n        <ol>{lis}\n        </ol>\n      </nav>\n'


def hero(h1: str, lead: str, items, fr: bool, updated: str) -> str:
    return (f'    <section class="page-hero" aria-labelledby="title">\n      <h1 id="title">{h1}</h1>\n'
            f'      <p class="lead">{lead}</p>\n      <p class="small faint">{updated}</p>\n'
            + toc(items, fr) + "    </section>\n")


def sec(i: int, sid: str, title: str, body: str) -> str:
    return (f'    <section class="sec doc" id="{sid}" aria-labelledby="h-{sid}">\n'
            f'      <h2 id="h-{sid}">{i:02d} · {title}</h2>\n{body}    </section>\n')


def p(*parts: str) -> str:
    return "".join(f"      <p>{x}</p>\n" for x in parts)


def table(head: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> str:
    th = "".join(f'<th scope="col"{" class=\"r\"" if j in right else ""}>{h}</th>' for j, h in enumerate(head))
    trs = "".join("<tr>" + "".join(f'<td{" class=\"r\"" if j in right else ""}>{c}</td>' for j, c in enumerate(r)) + "</tr>"
                  for r in rows)
    return f'      <div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>\n'


def ul(items: list[str]) -> str:
    return "      <ul>\n" + "".join(f"        <li>{x}</li>\n" for x in items) + "      </ul>\n"


TERMS_EN = [
    ("System One model", "A model that answers typed questions about a text in one forward pass, with a probability for each allowed answer. Also called a decision model or a Jev-class model."),
    ("Choice", "A question with a fixed list of options; the answer is one option and a probability for each."),
    ("Score", "A question on ordered levels (low, medium, high); the answer is a level and an expected value over the levels."),
    ("Noul", "Jev's name for a yes / no question: does the text support this statement? Kahn1 uses the same name."),
    ("Logit", "The raw score a language model gives each token before the softmax turns scores into probabilities."),
    ("Calibration", "How well probabilities match reality: of the answers given at 80% confidence, about 80% should be right."),
    ("ECE", "Expected calibration error: the average gap between confidence and accuracy, over bins of confidence. 0 is perfect."),
    ("Permutation debiasing", "Asking the same question with the options in k different orders and averaging the probabilities, so the position of an option does not sway the answer."),
    ("Prefix caching", "Reusing the computation of a shared prompt prefix (the state) across the questions asked about it, when the engine can: vLLM caches Kahn1 4B's prefix in 528-token blocks, so only states longer than a block benefit."),
]
TERMS_FR = [
    ("Modèle System One", "Un modèle qui répond à des questions typées sur un texte en une seule passe, avec une probabilité pour chaque réponse permise. On dit aussi modèle de décision ou modèle de classe Jev."),
    ("Choice", "Une question à liste d'options fixe ; la réponse est une option et une probabilité pour chacune."),
    ("Score", "Une question sur des niveaux ordonnés (faible, moyen, élevé) ; la réponse est un niveau et une espérance sur les niveaux."),
    ("Noul", "Le nom donné par Jev à une question oui / non : le texte appuie-t-il cette affirmation ? Kahn1 reprend ce nom."),
    ("Logit", "Le score brut qu'un modèle de langage donne à chaque token avant que le softmax en fasse des probabilités."),
    ("Calibration", "L'accord entre probabilités et réalité : sur les réponses données à 80 % de confiance, environ 80 % doivent être justes."),
    ("ECE", "Expected calibration error : l'écart moyen entre confiance et précision, par tranches de confiance. 0 est parfait."),
    ("Débiaisage par permutation", "Poser la même question avec les options dans k ordres différents et moyenner les probabilités, pour que la position d'une option ne pèse pas sur la réponse."),
    ("Cache de préfixe", "Réutiliser le calcul d'un début de prompt commun (l'état) d'une question à l'autre, quand le moteur le peut : vLLM met le préfixe de Kahn1 4B en cache par blocs de 528 tokens, seuls les états plus longs qu'un bloc en profitent."),
]


def pillar(fr: bool) -> tuple[str, list]:
    if fr:
        items = [("definition", "Définition"), ("fonctionnement", "Fonctionnement"), ("familles", "Les familles"),
                 ("choisir", "Comment choisir"), ("mesurer", "Comment on les mesure"), ("limites", "Limites"),
                 ("termes", "Termes clés")]
        h = hero("Modèles System One",
                 "Un modèle System One répond à des questions typées sur un texte en une seule passe, et renvoie "
                 "chaque réponse avec une probabilité pour chaque option permise. Jev, de "
                 "TypeSafe, a popularisé le terme en septembre 2026 ; des modèles ouverts font aujourd'hui la même chose.",
                 items, True, f"Mis à jour le 2 octobre 2026")
        b = [sec(1, "definition", "Définition", p(
            "Le nom vient du <em>Système 1</em> de Daniel Kahneman (<em>Système 1 / Système 2</em>, 2011) : des "
            "jugements rapides et bon marché qui règlent la plupart des cas, et laissent les cas difficiles à un "
            "Système 2 plus lent (un modèle de pointe ou une personne). TypeSafe écrit « System One » ; on lit aussi "
            "« System 1 », « modèle de décision » (decision model) et « modèle de classe Jev ».",
            "Un tel modèle reçoit un <em>état</em> (un message, un document, un ticket) et des questions typées. Il "
            "rend pour chacune une valeur typée et une distribution de probabilité : rien n'est généré, donc rien "
            "n'est à parser, et une réponse hors schéma ne peut pas arriver.") +
            table(["Type de question", "Ce qu'on pose", "Ce qui revient"], [
                ["Choice", "Une liste d'options fixe", "Une option + une probabilité par option"],
                ["Score", "Des niveaux ordonnés", "Un niveau + une espérance sur les niveaux"],
                ["Noul (oui / non)", "Une affirmation à vérifier", "Une probabilité que le texte l'appuie"]])),
             sec(2, "fonctionnement", "Fonctionnement", p(
                 "Le prompt contient l'état, la question et les options, chacune repérée par une lettre. Le modèle "
                 "fait une passe ; à la position de la réponse, on ne garde que les logits des tokens d'option (A, "
                 "B, C… ou oui / non), et le softmax les transforme en probabilités. Pour un Score, l'espérance "
                 "Σ i · p<sub>i</sub> donne une valeur continue.",
                 "Deux réglages rendent ces probabilités utilisables. Le <strong>débiaisage par permutation</strong> "
                 "pose la question avec les options dans k ordres et moyenne les résultats. La <strong>calibration</strong> "
                 "(une température par type de question) aligne la confiance sur la précision ; on la mesure par "
                 "l'ECE. Enfin, sur un état long, le <strong>cache de préfixe</strong> réutilise le calcul de l'état "
                 "d'une question à l'autre ; sur un état court, le moteur le recalcule pour chaque question, et la "
                 "latence croît avec le nombre de questions.") +
                 table(["", "Modèle System One", "LLM qui génère du JSON"], [
                     ["Sortie", "Une valeur typée avec sa distribution", "Du texte à valider et parser"],
                     ["Erreur de format", "Impossible par construction", "Rare avec structured outputs, possible sinon"],
                     ["Probabilités", "Une par option, toujours", "Logprobs selon le fournisseur, souvent aucune"],
                     ["Ce qui fait la latence", "Une passe par question, sur l'état", "La longueur de la réponse générée"],
                     ["Ce qui fait le coût", "Les tokens d'entrée", "Les tokens d'entrée et de sortie"]])),
             sec(3, "familles", "Les familles", p(
                 "Sept familles donnent une décision typée à partir d'un texte ; seuls les modèles de décision "
                 "combinent un format garanti, une probabilité par option et une seule passe. Le détail, produit par "
                 "produit, avec licences et sources, est sur la page <a href=\"/fr/comparer/paysage/\">Paysage</a>.") +
                 table(["Famille", "Exemples", "Licence", "Où ça tourne"], [
                     ["Modèles de décision hébergés", "Jev (TypeSafe), d1 (Liquid AI), Decisions API (OpenAI, accès anticipé limité)", "Fermée", "API"],
                     ["Modèles de décision ouverts", "Clef (Cloudflare), JevK5, Kahn1, Tev1 (Together), Laya", "Apache 2.0 le plus souvent", "Local, parfois aussi hébergé"],
                     ["LLM avec structured outputs", "OpenAI, Anthropic, Google", "Fermée", "API"],
                     ["Décodage contraint", "Outlines, XGrammar, vLLM, llama.cpp, Guidance", "Apache 2.0 ou MIT", "Local, avec n'importe quel modèle"],
                     ["Classification cloud", "Google Natural Language, AWS Comprehend, Azure Language", "Fermée", "API"],
                     ["Encodeurs zero-shot", "bart-large-mnli, DeBERTa-v3 zeroshot, GLiClass, SetFit", "MIT ou Apache 2.0", "CPU"],
                     ["Guardrails et juges", "Llama Guard 4, ShieldGemma, Granite Guardian, Qwen3Guard, Prometheus 2", "Mixte", "Local"]])),
             sec(4, "choisir", "Comment choisir", ul([
                 "<strong>Vos données peuvent-elles sortir ?</strong> Non : un modèle ouvert en local. Oui : une API hébergée devient possible.",
                 "<strong>Quelle langue ?</strong> Jev est entraîné d'abord pour l'anglais, selon sa documentation ; vérifiez la langue de chaque modèle.",
                 "<strong>Quelle licence ?</strong> Apache 2.0 et MIT permettent l'usage commercial ; certaines licences (Qwen Research, CC BY-NC) le limitent.",
                 "<strong>La calibration tient-elle sur vos données ?</strong> Mesurez l'ECE sur quelques centaines d'exemples annotés avant de fixer un seuil.",
                 "<strong>Combien d'options et quelle longueur de texte ?</strong> Le nombre d'options et la fenêtre de contexte varient d'un modèle à l'autre.",
                 "<strong>Quel débit et quel matériel ?</strong> Un GPU pour la production ; un CPU suffit pour tester, à quelques secondes par question.",
                 "<strong>Un encodeur suffit-il ?</strong> Pour des intentions courtes et fixes, un encodeur zero-shot sur CPU est souvent plus rapide et assez bon."])),
             sec(5, "mesurer", "Comment on les mesure", p(
                 "<a href=\"https://github.com/fstandhartinger/jevbench\" rel=\"noopener\">JevBench</a> est le benchmark "
                 "indépendant de la catégorie : des états et des grilles bornées, une réponse typée, quatre axes "
                 "(précision, calibration, vitesse, coût). Les éditeurs citent aussi leurs propres indices : Cloudflare et "
                 "Liquid AI annoncent chacun dépasser Jev sur un « Decision Index », chiffres non reproduits ici.",
                 f"Une comparaison n'a de sens qu'à périmètre égal : les mêmes exemples, les mêmes options, les mêmes "
                 f"labels. C'est la règle de nos <a href=\"/fr/resultats/\">résultats</a> : sur 14{N}663 exemples réservés, "
                 f"Jev 1.13.0 obtient 73,2{N}% et Kahn1 4B 72,4{N}%. Sur les 231 exemples publics de JevBench, Jev obtient "
                 f"86,6{N}% dans l'exécution de JevBench, Kahn1 4B 87,4{N}% dans la nôtre (k{N}={N}3, calibré ; écart non "
                 f"significatif, p{N}={N}0,84) et JevK5 v0.2 "
                 f"85,3{N}% dans celle de JevBench (86,1{N}% dans celle de ses auteurs) : trois exécutions distinctes sur les "
                 f"mêmes exemples.")),
             sec(6, "limites", "Limites", ul([
                 "Ce ne sont pas des modèles de raisonnement : calculer une date, une durée ou un montant en une passe reste un point faible.",
                 "La calibration vaut pour la distribution sur laquelle elle a été ajustée : recalibrez sur votre domaine.",
                 "La confiance n'est pas la probabilité d'avoir raison : fixez vos seuils sur vos propres données.",
                 "La réglementation s'applique toujours (AI Act, règles sectorielles) : gardez une personne ou un modèle plus grand sur les cas peu sûrs."]) +
                 p("Le détail, pour Kahn1, est sur la page <a href=\"/fr/vigilance/\">Points de vigilance</a>.")),
             sec(7, "termes", "Termes clés", "      <ul class=\"plain\">\n" + "".join(
                 f"        <li><b>{t}.</b> {d}</li>\n" for t, d in TERMS_FR) + "      </ul>\n")]
        return h + "".join(b), TERMS_FR
    items = [("definition", "Definition"), ("how", "How it works"), ("families", "The families"),
             ("choose", "How to choose"), ("measure", "How they are measured"), ("limits", "Limits"),
             ("terms", "Key terms")]
    h = hero("System One models",
             "A System One model answers typed questions about a text in one forward pass, and returns each answer "
             "with a probability for every allowed option. TypeSafe's Jev made the term popular "
             "in September 2026; open models now do the same.", items, False, "Updated October 2, 2026")
    b = [sec(1, "definition", "Definition", p(
        "The name comes from Daniel Kahneman's <em>System 1</em> (<em>Thinking, Fast and Slow</em>, 2011): fast, cheap "
        "judgments that settle most cases and leave the hard ones to a slower System 2 (a frontier model or a person). "
        "TypeSafe writes “System One”; you will also read “System 1”, “decision model” and "
        "“Jev-class model”.",
        "Such a model takes a <em>state</em> (a message, a document, a ticket) and typed questions. For each it returns a "
        "typed value and a probability distribution: nothing is generated, so nothing has to be parsed, and an answer "
        "outside the schema cannot happen.") +
        table(["Question type", "What you ask", "What comes back"], [
            ["Choice", "A fixed list of options", "One option + a probability for each"],
            ["Score", "Ordered levels", "A level + an expected value over the levels"],
            ["Noul (yes / no)", "A statement to check", "The probability that the text supports it"]])),
         sec(2, "how", "How it works", p(
             "The prompt holds the state, the question and the options, each tagged with a letter. The model runs one "
             "forward pass; at the answer position, only the logits of the option tokens (A, B, C… or yes / no) are "
             "kept, and the softmax turns them into probabilities. For a Score, the expected value Σ i · p<sub>i</sub> "
             "gives a continuous value.",
             "Two settings make those probabilities usable. <strong>Permutation debiasing</strong> asks the question "
             "with the options in k orders and averages the results. <strong>Calibration</strong> (one temperature per "
             "question type) aligns confidence with accuracy, measured by the ECE. On a long state, <strong>prefix "
             "caching</strong> also reuses the state's computation from one question to the next; on a short state the "
             "engine recomputes it for each question, and latency grows with the number of questions.") +
             table(["", "System One model", "LLM generating JSON"], [
                 ["Output", "A typed value with its distribution", "Text to validate and parse"],
                 ["Format errors", "Impossible by construction", "Rare with structured outputs, possible otherwise"],
                 ["Probabilities", "One per option, always", "Logprobs depending on the provider, often none"],
                 ["What drives latency", "One pass per question, over the state", "The length of the generated answer"],
                 ["What drives cost", "Input tokens", "Input and output tokens"]])),
         sec(3, "families", "The families", p(
             "Seven families turn a text into a typed decision; only decision models combine a guaranteed format, a "
             "probability per option and a single pass. Product by product, with licences and sources, see the "
             "<a href=\"/compare/landscape/\">landscape</a>.") +
             table(["Family", "Examples", "Licence", "Where it runs"], [
                 ["Hosted decision models", "Jev (TypeSafe), d1 (Liquid AI), Decisions API (OpenAI, limited preview)", "Closed", "API"],
                 ["Open decision models", "Clef (Cloudflare), JevK5, Kahn1, Tev1 (Together), Laya", "Mostly Apache 2.0", "Local, sometimes also hosted"],
                 ["LLMs with structured outputs", "OpenAI, Anthropic, Google", "Closed", "API"],
                 ["Constrained decoding", "Outlines, XGrammar, vLLM, llama.cpp, Guidance", "Apache 2.0 or MIT", "Local, with any model"],
                 ["Cloud classification", "Google Natural Language, AWS Comprehend, Azure Language", "Closed", "API"],
                 ["Zero-shot encoders", "bart-large-mnli, DeBERTa-v3 zeroshot, GLiClass, SetFit", "MIT or Apache 2.0", "CPU"],
                 ["Guard and judge models", "Llama Guard 4, ShieldGemma, Granite Guardian, Qwen3Guard, Prometheus 2", "Mixed", "Local"]])),
         sec(4, "choose", "How to choose", ul([
             "<strong>Can your data leave your infrastructure?</strong> No: an open model, run locally. Yes: a hosted API becomes an option.",
             "<strong>Which language?</strong> Jev is trained first for English, per its documentation; check the language of each model.",
             "<strong>Which licence?</strong> Apache 2.0 and MIT allow commercial use; some licences (Qwen Research, CC BY-NC) limit it.",
             "<strong>Does calibration hold on your data?</strong> Measure the ECE on a few hundred labelled examples before you set a threshold.",
             "<strong>How many options, how long a text?</strong> The number of options and the context window differ from one model to the next.",
             "<strong>What throughput, what hardware?</strong> A GPU for production; a CPU is enough to try it, at seconds per question.",
             "<strong>Is an encoder enough?</strong> For short, fixed intents, a zero-shot encoder on a CPU is often faster and good enough."])),
         sec(5, "measure", "How they are measured", p(
             "<a href=\"https://github.com/fstandhartinger/jevbench\" rel=\"noopener\">JevBench</a> is the independent "
             "benchmark of the category: states and bounded rubrics, a typed answer, four axes (accuracy, calibration, "
             "speed, cost). Vendors also cite their own indices: Cloudflare and Liquid AI each report beating Jev on a "
             "“Decision Index”, figures not reproduced here.",
             "A comparison only means something like for like: the same items, the same options, the same labels. That "
             "is the rule of our <a href=\"/benchmarks/\">benchmarks</a>: on 14,663 held-out items, Jev 1.13.0 scores "
             "73.2% and Kahn1 4B 72.4%. On the 231 public JevBench items, Jev scores 86.6% in JevBench's run, Kahn1 4B "
             "87.4% in ours (k&nbsp;=&nbsp;3, calibrated; not a significant gap, p&nbsp;=&nbsp;0.84) and JevK5 v0.2 85.3% "
             "in JevBench's run (86.1% in its authors' "
             "own run): three separate runs on the same items.")),
         sec(6, "limits", "Limits", ul([
             "They are not reasoning models: computing a date, a duration or an amount in one pass stays a weak spot.",
             "Calibration holds on the distribution it was fitted on: recalibrate on your domain.",
             "Confidence is not the probability of being right: set thresholds on your own data.",
             "Regulation still applies (EU AI Act, sector rules): keep a person or a larger model on the low-confidence path."]) +
             p("For Kahn1 in detail, see the <a href=\"/caveats/\">caveats</a>.")),
         sec(7, "terms", "Key terms", "      <ul class=\"plain\">\n" + "".join(
             f"        <li><b>{t}.</b> {d}</li>\n" for t, d in TERMS_EN) + "      </ul>\n")]
    return h + "".join(b), TERMS_EN


def landscape(fr: bool) -> str:
    fams = list(FAMILIES.items())
    if fr:
        items = [("apercu", "En bref")] + [(f"f-{k}", v[1]) for k, v in fams] + [("methode", "Méthode et données")]
        h = hero("Paysage des modèles de décision",
                 "Sept familles de solutions transforment un texte en décision typée, propriétaires et ouvertes. Cette "
                 f"page les recense produit par produit, avec licence, déploiement, sortie, probabilités, prix et source : "
                 f"{len(ROWS)} entrées, chacune liée à sa source, relevées le 1er octobre 2026.", items, True, "Mis à jour le 2 octobre 2026")
        intro = sec(1, "apercu", "En bref", p(
            "Seuls les modèles de décision, hébergés ou ouverts, combinent un format garanti, une probabilité par option "
            "et une seule passe par question. Les LLM avec structured outputs garantissent le format mais pas les "
            "probabilités ; les encodeurs zero-shot donnent des probabilités, sur CPU, mais sans calibration annoncée.",
            "Les chiffres de performance publiés par chacun viennent de jeux différents : ils ne se comparent pas entre "
            "eux. Nos comparaisons à périmètre égal sont sur la page <a href=\"/fr/resultats/\">Résultats</a>."))
        meth = sec(len(fams) + 2, "methode", "Méthode et données", p(
            "Chaque ligne renvoie, depuis son nom, à la page dont elle vient : celle de l'éditeur ou son dépôt, "
            "relevée le 1er octobre 2026. Les "
            "affirmations des éditeurs (calibration, vitesse) sont rapportées comme telles. Une case vide signifie "
            "que l'information n'était pas sur la page.",
            "Les données sont téléchargeables : <a href=\"/data/landscape.csv\">landscape.csv</a> (en anglais). D'autres "
            "projets cités par des annuaires tiers (Kev, Von, NanoJev, SemIf, djev…) entreront ici quand on aura pu les "
            "vérifier sur leurs propres pages. Une erreur ou un oubli : "
            "<a href=\"https://github.com/Okura66/kahn1/issues\" rel=\"noopener\">ouvrez une issue</a>."))
    else:
        items = [("overview", "At a glance")] + [(f"f-{k}", v[0]) for k, v in fams] + [("method", "Method and data")]
        h = hero("Decision model landscape",
                 "Seven families of tools turn a text into a typed decision, proprietary and open. This page lists them "
                 f"product by product, with licence, deployment, output, probabilities, price and source: {len(ROWS)} "
                 "entries, each linked to its source, checked on October 1, 2026.", items, False, "Updated October 2, 2026")
        intro = sec(1, "overview", "At a glance", p(
            "Only decision models, hosted or open, combine a guaranteed format, a probability per option and a single "
            "pass per question. LLMs with structured outputs guarantee the format but not the probabilities; "
            "zero-shot encoders give probabilities, on a CPU, but make no calibration claim.",
            "The performance figures each provider publishes come from different test sets: they do not compare with "
            "each other. Our like-for-like comparisons are on the <a href=\"/benchmarks/\">benchmarks</a> page."))
        meth = sec(len(fams) + 2, "method", "Method and data", p(
            "Each row links, from its name, to the page it comes from: the provider's own page or repository, "
            "checked on October 1, 2026. Vendor "
            "claims (calibration, speed) are reported as claims. An empty cell means the page did not say.",
            "Download the data: <a href=\"/data/landscape.csv\">landscape.csv</a>. Other projects listed by third-party "
            "directories (Kev, Von, NanoJev, SemIf, djev…) will join once we can check them on their own pages. Found an "
            "error or a gap? <a href=\"https://github.com/Okura66/kahn1/issues\" rel=\"noopener\">Open an issue</a>."))
    return h + intro + landscape_tables(fr) + meth


FAQ_EN = [
    ("Is Jev open source?", "No. Jev is a hosted API from TypeSafe; its weights are not published. You call it at api.typesafe.ai and pay per input token ($0.042 per million, output free, per its documentation)."),
    ("Can I run Jev locally?", "Not Jev itself. Open decision models run on your own hardware: Clef and Clef-flash (Cloudflare), JevK5, Laya, Kahn1 and others, several of them under Apache 2.0."),
    ("Which open alternative is closest to Jev?", "It depends on the task, and no open model has been measured as a drop-in replacement on every task. In JevBench's own runs on the 231 public items (v1.4), several systems with public code or weights score at or above Jev's 86.6%: NInfer Qwen3.8-Flash-Next and JevOne (89.6%), OpenJev (thinking) and swanOne (88.7%), djev (thinking, 87.4%), reflex-27b (87.0%) and SimpleJev Qwen3.8-27B (86.6%); JevK5 v0.2 scores 85.3%. Kahn1 4B scores 87.4% on the same items in our own run, which JevBench has not reproduced; paired against Jev's published outcomes the gap is not significant (exact McNemar p = 0.84). On Kahn1's 14,663 held-out items, Jev 1.13.0 scores 73.2% and Kahn1 4B 72.4%."),
    ("Which alternatives accept Jev's request format?", "Clef is announced as fully Jev-API compatible, and JevK5 and Laya (through laya-serve) serve the POST /v1/systemone request shape. Kahn1 takes the same question fields (type, instructions, criteria) under a \"schema\" key at POST /v1/evaluate/jev, so a Jev client needs a small adapter."),
    ("Is Kahn1 affiliated with TypeSafe?", "No. Kahn1 is an independent project; it takes Jev's question fields and compares itself with Jev on the same items."),
]
FAQ_FR = [
    ("Jev est-il open source ?", "Non. Jev est une API hébergée par TypeSafe ; ses poids ne sont pas publiés. On l'appelle sur api.typesafe.ai et on paie les tokens d'entrée (0,042 $ par million, sortie gratuite, selon sa documentation)."),
    ("Peut-on faire tourner Jev en local ?", "Pas Jev lui-même. Des modèles de décision ouverts tournent sur votre matériel : Clef et Clef-flash (Cloudflare), JevK5, Laya, Kahn1 et d'autres, plusieurs sous Apache 2.0."),
    ("Quelle alternative ouverte est la plus proche de Jev ?", "Cela dépend de la tâche, et aucun modèle ouvert n'a été mesuré comme remplaçant direct sur toutes les tâches. Dans les exécutions de JevBench lui-même sur les 231 exemples publics (v1.4), plusieurs systèmes à code ou poids publics atteignent ou dépassent les 86,6 % de Jev : NInfer Qwen3.8-Flash-Next et JevOne (89,6 %), OpenJev (thinking) et swanOne (88,7 %), djev (thinking, 87,4 %), reflex-27b (87,0 %) et SimpleJev Qwen3.8-27B (86,6 %) ; JevK5 v0.2 obtient 85,3 %. Kahn1 4B obtient 87,4 % sur les mêmes exemples dans notre propre exécution, que JevBench n'a pas reproduite ; apparié aux résultats publiés de Jev, l'écart n'est pas significatif (test exact de McNemar, p = 0,84). Sur les 14 663 exemples réservés de Kahn1, Jev 1.13.0 obtient 73,2 % et Kahn1 4B 72,4 %."),
    ("Quelles alternatives acceptent le format de requête de Jev ?", "Clef est annoncé entièrement compatible avec l'API Jev, et JevK5 et Laya (via laya-serve) servent des requêtes au format POST /v1/systemone. Kahn1 prend les mêmes champs de question (type, instructions, criteria) sous une clé « schema » sur POST /v1/evaluate/jev : un client Jev demande donc un petit adaptateur."),
    ("Kahn1 est-il affilié à TypeSafe ?", "Non. Kahn1 est un projet indépendant ; il prend les champs de question de Jev et se compare à Jev sur les mêmes exemples."),
]


def alternatives(fr: bool) -> tuple[str, list]:
    if fr:
        items = [("reponse", "La réponse courte"), ("alternatives", "Les alternatives ouvertes"), ("mesures", "Ce qui est mesuré"),
                 ("migrer", "Passer de Jev à un modèle ouvert"), ("jev", "Quand Jev reste le bon choix"), ("faq", "Questions fréquentes")]
        h = hero("Alternatives open source à Jev",
                 "Oui, Jev a des alternatives ouvertes : plusieurs modèles de décision à poids ouverts répondent au même "
                 "type de questions typées sur votre propre matériel, et quatre prennent des questions au format de Jev : "
                 "Clef, JevK5 et Laya servent son format de requête, Kahn1 prend les mêmes champs de question sur sa propre "
                 "route. Aucun n'est mesuré comme remplaçant direct sur toutes les tâches.", items, True, "Mis à jour le 2 octobre 2026")
        alt = table(["Alternative", "Éditeur", "Licence", "Tailles", "Compatible Jev", "Mesurée ici à périmètre égal"], [
            ['<a href="https://blog.cloudflare.com/clef-decision-models/" rel="noopener">Clef, Clef-flash</a>', "Cloudflare", "Apache 2.0", "27B, 9B", "Oui, annoncé (API Jev)", "Pas encore"],
            ['<a href="https://github.com/allebee/jevk5" rel="noopener">JevK5 v0.3</a>', "allebee", "Apache 2.0", "4B, 9B (+ 2B, Lite)", "Format /v1/systemone", "v0.2 sur JevBench public (exécution de ses auteurs et de JevBench)"],
            ['<a href="https://huggingface.co/convaiinnovations/laya" rel="noopener">Laya</a>', "Convai Innovations", "Apache 2.0", "421M, 322M (encodeurs)", "Format /v1/systemone (laya-serve)", "Non"],
            ['<a href="/fr/modeles/">Kahn1</a>', "Kahn1", "4B Apache 2.0, 3B licence de recherche Qwen", "4B, 3B", "Champs de question de Jev sous une clé « schema » sur /v1/evaluate/jev ; pas un remplaçant direct pour un client Jev", "Oui : holdout et JevBench public"],
            ['<a href="https://github.com/ikermoel/open-alternative-jev" rel="noopener">open-alternative-jev (so1)</a>', "ikermoel", "Apache 2.0", "N'importe quel LLM ouvert", "", "Non"],
            ['<a href="https://huggingface.co/togethercomputer/Tev1-4B-experimental" rel="noopener">Tev1-4B-experimental</a>', "Together AI", "En cours", "4B", "", "Non"]])
        mes = table(["", "Exemples", "Jev 1.13.0", "JevK5 v0.2, exécution JevBench", "JevK5 v0.2, exécution de ses auteurs", "Kahn1 4B"], [
            ["Holdout Kahn1, global (Choice sur les mêmes 8 options)", f"14{N}663", f"<b>73,2{N}%</b>", "", "", f"72,4{N}%"],
            ["JevBench public, tous niveaux", "231", f"86,6{N}%", f"85,3{N}%", f"86,1{N}%", f"<b>87,4{N}%</b>"],
            ["JevBench public, niveau difficile", "111", f"73,0{N}%", "", f"73,9{N}%", f"<b>75,7{N}%</b>"]], {1, 2, 3, 4, 5})
        body = [sec(1, "reponse", "La réponse courte", p(
            "Si vous voulez des décisions typées avec probabilités sans envoyer vos données à une API, prenez un modèle "
            "de décision ouvert. Si vous voulez la meilleure précision globale mesurée sans rien héberger, Jev reste devant sur "
            f"notre holdout : 73,2{N}% contre 72,4{N}% pour Kahn1 4B sur 14{N}663 exemples réservés. Sur les 231 exemples "
            f"publics de JevBench, les deux sont au même niveau (Kahn1 4B 87,4{N}%, Jev 86,6{N}%, écart non significatif, p{N}={N}0,84).")),
                sec(2, "alternatives", "Les alternatives ouvertes", alt + p(
                    "Le paysage complet, hébergé et ouvert, est sur la page <a href=\"/fr/comparer/paysage/\">Paysage</a>.")),
                sec(3, "mesures", "Ce qui est mesuré", mes + p(
                    "Mêmes exemples, mêmes options, mêmes labels, mais trois exécutions distinctes : les résultats de Jev "
                    "sont ceux que publie JevBench, ceux de JevK5 viennent de l'exécution publiée par ses auteurs et de celle "
                    f"de JevBench, ceux de Kahn1 de la nôtre (k{N}={N}3 ordres d'options, calibré). Apparié à l'exécution "
                    f"publiée par les auteurs de JevK5, Kahn1 4B réussit 202 exemples sur 231 et JevK5 199 (13 contre 10 exemples "
                    f"que seul l'un réussit, test exact de McNemar, p{N}={N}0,68) ; face aux résultats publiés de Jev (200 sur 231), "
                    f"13 contre 11, p{N}={N}0,84 ; dans l'exécution de JevBench, JevK5 v0.2 en réussit 197 (85,3{N}%). "
                    "Détails : <a href=\"/fr/resultats/\">Résultats</a>.")),
                sec(4, "migrer", "Passer de Jev à un modèle ouvert", p(
                    "Les trois types de question se retrouvent tels quels : choice, score et noul. Clef, JevK5 et Laya "
                    "servent le format de requête de Jev, <code>POST /v1/systemone</code>. Kahn1 prend les mêmes champs de "
                    "question (<code>type</code>, <code>instructions</code>, <code>criteria</code>) sous une clé "
                    "<code>schema</code> sur <code>POST /v1/evaluate/jev</code>, alors qu'un client Jev envoie "
                    "<code>questions</code> et <code>model</code> à <code>/v1/systemone</code> : prévoyez un petit "
                    "adaptateur. Recalibrez sur quelques centaines de vos exemples avant de reprendre les seuils réglés pour Jev.") +
                    '      <pre class="term"><span class="d">$ </span>SYSONE_MODEL=Okura66/Kahn1-Qwen3.5-4B uv run sysone serve --port 8000\n'
                    '<span class="d">$ </span>curl -X POST http://127.0.0.1:8000/v1/evaluate/jev -H "Content-Type: application/json" \\\n'
                    '    -d \'{"state": "Bonjour, impossible de me connecter depuis ce matin.", "schema": {"categorie": {"type": "choice",\n'
                    '         "instructions": "Catégorie du ticket", "criteria": {"bug": "Une erreur", "compte": "Connexion, accès"}}}}\'</pre>\n'),
                sec(5, "jev", "Quand Jev reste le bon choix", ul([
                    "Vous voulez la meilleure précision globale mesurée sur notre holdout, sans infrastructure à gérer.",
                    "Vos textes sont longs : Jev accepte 64k tokens par requête (32k pour l'état et la plus longue question), selon sa documentation.",
                    "Vous travaillez en anglais, la langue où Jev est le plus précis selon TypeSafe.",
                    "Vous acceptez qu'un tiers traite vos données et un prix par token d'entrée."])),
                sec(6, "faq", "Questions fréquentes", "".join(
                    f"      <h3 class=\"sub\">{q}</h3>\n      <p>{esc(a)}</p>\n" for q, a in FAQ_FR))]
        return h + "".join(body), FAQ_FR
    items = [("answer", "The short answer"), ("alternatives", "The open alternatives"), ("measured", "What is measured"),
             ("switch", "Switching from Jev"), ("jev", "When Jev is still the right pick"), ("faq", "FAQ")]
    h = hero("Open-source alternatives to Jev",
             "Yes, Jev has open alternatives: several open-weight decision models answer the same kind of typed questions "
             "on your own hardware, and four take Jev-style questions: Clef, JevK5 and Laya serve its request format, and "
             "Kahn1 takes the same question fields on its own route. None has been measured as a drop-in replacement on "
             "every task.", items, False, "Updated October 2, 2026")
    alt = table(["Alternative", "Provider", "Licence", "Sizes", "Jev-compatible", "Measured here like for like"], [
        ['<a href="https://blog.cloudflare.com/clef-decision-models/" rel="noopener">Clef, Clef-flash</a>', "Cloudflare", "Apache 2.0", "27B, 9B", "Yes, announced (Jev API)", "Not yet"],
        ['<a href="https://github.com/allebee/jevk5" rel="noopener">JevK5 v0.3</a>', "allebee", "Apache 2.0", "4B, 9B (+ 2B, Lite)", "/v1/systemone request shape", "v0.2 on public JevBench (its authors' run and JevBench's)"],
        ['<a href="https://huggingface.co/convaiinnovations/laya" rel="noopener">Laya</a>', "Convai Innovations", "Apache 2.0", "421M, 322M (encoders)", "/v1/systemone request shape (laya-serve)", "No"],
        ['<a href="/models/">Kahn1</a>', "Kahn1", "4B Apache 2.0, 3B Qwen Research License", "4B, 3B", "Jev's question fields under a “schema” key at /v1/evaluate/jev; not a drop-in for Jev clients", "Yes: held-out and public JevBench"],
        ['<a href="https://github.com/ikermoel/open-alternative-jev" rel="noopener">open-alternative-jev (so1)</a>', "ikermoel", "Apache 2.0", "Any open LLM", "", "No"],
        ['<a href="https://huggingface.co/togethercomputer/Tev1-4B-experimental" rel="noopener">Tev1-4B-experimental</a>', "Together AI", "Being finalized", "4B", "", "No"]])
    mes = table(["", "Items", "Jev 1.13.0", "JevK5 v0.2, JevBench's run", "JevK5 v0.2, its authors' run", "Kahn1 4B"], [
        ["Kahn1 held-out, all primitives (Choice over the same 8 options)", "14,663", "<b>73.2%</b>", "", "", "72.4%"],
        ["Public JevBench, all tiers", "231", "86.6%", "85.3%", "86.1%", "<b>87.4%</b>"],
        ["Public JevBench, hard tier", "111", "73.0%", "", "73.9%", "<b>75.7%</b>"]], {1, 2, 3, 4, 5})
    body = [sec(1, "answer", "The short answer", p(
        "If you want typed decisions with probabilities without sending your data to an API, pick an open decision "
        "model. If you want the best measured overall accuracy with nothing to host, Jev is still ahead on our held-out set: "
        "73.2% against 72.4% for Kahn1 4B on 14,663 items. On the 231 public JevBench items the two are level (Kahn1 4B "
        "87.4%, Jev 86.6%, not a significant gap, p&nbsp;=&nbsp;0.84).")),
            sec(2, "alternatives", "The open alternatives", alt + p(
                "The full landscape, hosted and open, is on the <a href=\"/compare/landscape/\">landscape</a> page.")),
            sec(3, "measured", "What is measured", mes + p(
                "Same items, same options, same labels, but three separate runs: Jev's outcomes are the ones JevBench "
                "publishes, JevK5's come from its authors' published run and from JevBench's own run, and Kahn1's from ours "
                "(k&nbsp;=&nbsp;3 option orders, calibrated). Paired with JevK5's own published run, Kahn1 4B gets 202 of 231 "
                "items right and JevK5 199 (13 against 10 items only one gets right, exact McNemar test, p&nbsp;=&nbsp;0.68); "
                "against Jev's published outcomes (200 of 231), 13 against 11, p&nbsp;=&nbsp;0.84; in JevBench's own run, "
                "JevK5 v0.2 gets 197 (85.3%). Details: <a href=\"/benchmarks/\">benchmarks</a>.")),
            sec(4, "switch", "Switching from Jev", p(
                "The three question types carry over as they are: choice, score and noul. Clef, JevK5 and Laya serve Jev's "
                "request format, <code>POST /v1/systemone</code>. Kahn1 takes the same question fields (<code>type</code>, "
                "<code>instructions</code>, <code>criteria</code>) under a <code>schema</code> key at "
                "<code>POST /v1/evaluate/jev</code>, while a Jev client sends <code>questions</code> and <code>model</code> "
                "to <code>/v1/systemone</code>: plan a small adapter. Recalibrate on a few hundred of your own examples "
                "before reusing thresholds tuned for Jev.") +
                '      <pre class="term"><span class="d">$ </span>SYSONE_MODEL=Okura66/Kahn1-Qwen3.5-4B uv run sysone serve --port 8000\n'
                '<span class="d">$ </span>curl -X POST http://127.0.0.1:8000/v1/evaluate/jev -H "Content-Type: application/json" \\\n'
                '    -d \'{"state": "Hello, I cannot log in to my account.", "schema": {"category": {"type": "choice",\n'
                '         "instructions": "Support ticket category", "criteria": {"bug": "Something is broken", "account": "Login, access"}}}}\'</pre>\n'),
            sec(5, "jev", "When Jev is still the right pick", ul([
                "You want the best measured overall accuracy on our held-out set, with no infrastructure to run.",
                "Your texts are long: Jev takes 64k tokens per request (32k for the state plus the longest question), per its documentation.",
                "You work in English, where TypeSafe says Jev is most accurate.",
                "You are fine with a third party processing your data and a price per input token."])),
            sec(6, "faq", "FAQ", "".join(f"      <h3 class=\"sub\">{q}</h3>\n      <p>{esc(a)}</p>\n" for q, a in FAQ_EN))]
    return h + "".join(body), FAQ_EN


# ---------------------------------------------------------------------------------------------
# Pages: path EN, path FR, titles, descriptions, builders
# ---------------------------------------------------------------------------------------------
PAGES = [
    dict(key="pillar", en="/learn/system-one-models/", fr="/fr/apprendre/modeles-system-one/",
         title_en="System One models: what they are and how to choose",
         title_fr="Modèles System One : définition, fonctionnement, choix",
         desc_en="What a System One (decision) model is, how it reads typed answers from logits, the families, "
                 "proprietary and open, and how to choose one. Like-for-like figures.",
         desc_fr="Ce qu'est un modèle System One (modèle de décision), comment il lit des réponses typées dans les "
                 "logits, familles propriétaires et ouvertes, comment choisir.",
         crumb_en="System One models", crumb_fr="Modèles System One", nav_en="System One models", nav_fr="Modèles System One"),
    dict(key="landscape", en="/compare/landscape/", fr="/fr/comparer/paysage/",
         title_en="Decision model landscape: proprietary vs open source",
         title_fr="Paysage des modèles de décision : propriétaires et ouverts",
         desc_en="Every way to turn text into a typed decision, proprietary and open: decision models, LLMs, "
                 "decoding libraries, encoders, guards. Licence, output, price, source.",
         desc_fr="Toutes les façons de tirer une décision typée d'un texte, propriétaires et ouvertes : modèles de "
                 "décision, LLM, encodeurs, guardrails. Licence, prix, source.",
         crumb_en="Landscape", crumb_fr="Paysage", nav_en="Landscape", nav_fr="Paysage"),
    dict(key="alternatives", en="/alternatives/jev/", fr="/fr/alternatives/jev/",
         title_en="Open-source alternatives to Jev, compared like for like",
         title_fr="Alternatives open source à Jev, comparées à périmètre égal",
         desc_en="Open-weight alternatives to TypeSafe's Jev: Clef, JevK5, Kahn1 and more. Licences, Jev-compatible "
                 "APIs, like-for-like accuracy, when Jev is still the pick.",
         desc_fr="Les alternatives à poids ouverts au Jev de TypeSafe : Clef, JevK5, Kahn1 et d'autres. Licences, API "
                 "compatibles, précision à périmètre égal, quand garder Jev.",
         crumb_en="JEV alternatives", crumb_fr="Alternatives à JEV", nav_en="JEV alternatives", nav_fr="Alternatives à JEV"),
]

KW = {
    "pillar": ("System One model, System 1 model, decision model, Jev-class model, typed decisions, logit-based classification, calibrated LLM classifier",
               "modèle System One, modèle Système 1, modèle de décision, décisions typées, classification par logits, classifieur LLM calibré"),
    "landscape": ("decision models compared, JEV alternatives, structured outputs vs decision models, zero-shot classification models, guardrail models, LLM as a judge",
                  "comparatif modèles de décision, alternatives à JEV, structured outputs, classification zero-shot, modèles guardrails, LLM juge"),
    "alternatives": ("open source JEV alternative, JEV alternative, run JEV locally, JEV vs JevK5, JEV vs Kahn1, Clef, JevK5, Kahn1",
                     "alternative open source à JEV, alternative à JEV, JEV en local, JEV ou JevK5, Clef, JevK5, Kahn1"),
}


def fr_typo(html: str) -> str:
    """French typography on visible text: no-break space before : ; ! ? % and inside « », outside tags and code."""
    parts = re.split(r"(<pre.*?</pre>|<code>.*?</code>|<[^>]+>)", html, flags=re.S)
    for i in range(0, len(parts), 2):
        s = re.sub(r" ([:;!?%»])", r"&nbsp;\1", parts[i])
        parts[i] = s.replace("« ", "«&nbsp;")
    return "".join(parts)


def build_page(tpl: str, page: dict, fr: bool, main_html: str, extra_ld: list[dict]) -> str:
    t = tpl.replace("\r\n", "\n")
    path, other = (page["fr"], page["en"]) if fr else (page["en"], page["fr"])
    url = SITE + path
    title, desc = (page["title_fr"], page["desc_fr"]) if fr else (page["title_en"], page["desc_en"])
    assert len(title) <= 60 and len(desc) <= 160, (path, len(title), len(desc))
    t = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", t, count=1)
    t = re.sub(r'<meta name="description" content="[^"]*">', f'<meta name="description" content="{esc(desc)}">', t, count=1)
    t = re.sub(r'<link rel="canonical" href="[^"]*">', f'<link rel="canonical" href="{url}">', t, count=1)
    en_url, fr_url = SITE + page["en"], SITE + page["fr"]
    t = re.sub(r'<link rel="alternate" hreflang="en" href="[^"]*">', f'<link rel="alternate" hreflang="en" href="{en_url}">', t, count=1)
    t = re.sub(r'<link rel="alternate" hreflang="fr" href="[^"]*">', f'<link rel="alternate" hreflang="fr" href="{fr_url}">', t, count=1)
    t = re.sub(r'<link rel="alternate" hreflang="x-default" href="[^"]*">', f'<link rel="alternate" hreflang="x-default" href="{en_url}">', t, count=1)
    t = re.sub(r'<meta property="og:url" content="[^"]*">', f'<meta property="og:url" content="{url}">', t, count=1)
    t = re.sub(r'<meta property="og:title" content="[^"]*">', f'<meta property="og:title" content="{esc(title)}">', t, count=1)
    t = re.sub(r'<meta property="og:description" content="[^"]*">', f'<meta property="og:description" content="{esc(desc)}">', t, count=1)
    t = re.sub(r'href="(?:\.\./)+assets/site\.css', 'href="/assets/site.css', t)
    # JSON-LD
    home = SITE + ("/fr/" if fr else "/")
    crumb = page["crumb_fr" if fr else "crumb_en"]
    art = {"@context": "https://schema.org", "@type": "TechArticle",
           "image": SITE + ("/assets/og-fr.png" if fr else "/assets/og.png"), "headline": title,
           "inLanguage": "fr" if fr else "en", "url": url, "description": desc,
           "keywords": KW[page["key"]][1 if fr else 0],
           "author": {"@type": "Person", "@id": SITE + "/#author", "name": "Axel Montzamir", "url": "https://www.linkedin.com/in/amontzamir/"},
           "publisher": {"@id": SITE + "/#author"},
           "about": {"@type": "SoftwareApplication", "@id": SITE + "/#software", "name": "Kahn1", "url": SITE + "/"},
           "isPartOf": {"@type": "WebSite", "@id": SITE + "/#website", "name": "Kahn1", "url": SITE + "/"},
           "datePublished": TODAY, "dateModified": MODIFIED}
    bc = {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": 1, "name": "Kahn1", "item": home},
        {"@type": "ListItem", "position": 2, "name": crumb, "item": url}]}
    blocks = [art, bc] + extra_ld
    ld = "\n".join(f'<script type="application/ld+json">\n{json.dumps(b, ensure_ascii=False, indent=1)}\n</script>' for b in blocks)
    t = re.sub(r'(<script type="application/ld\+json">\n.*?\n</script>\n?)+', ld + "\n", t, count=1, flags=re.S)
    # language switch and current page in the sidebar
    t = re.sub(r'(<div class="prefs"[^>]*>\s*<a href=")[^"]*(" hreflang="en")', lambda m: m[1] + page["en"] + m[2], t, count=1)
    t = re.sub(r'(<a href=")[^"]*(" hreflang="fr" lang="fr")', lambda m: m[1] + page["fr"] + m[2], t, count=1)
    t = t.replace(' aria-current="page">Caveats</a>', ">Caveats</a>").replace(' aria-current="page">Points de vigilance</a>', ">Points de vigilance</a>")
    # main
    if fr:
        main_html = fr_typo(main_html)
    t = re.sub(r"<main id=\"main\" class=\"page\">.*?</main>", f'<main id="main" class="page">\n{main_html}  </main>', t, count=1, flags=re.S)
    return t


def faq_ld(faq: list) -> dict:
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq]}


def terms_ld(terms: list, url: str, fr: bool) -> dict:
    return {"@context": "https://schema.org", "@type": "DefinedTermSet", "@id": url + "#terms",
            "name": "Termes clés des modèles System One" if fr else "System One model key terms",
            "hasDefinedTerm": [{"@type": "DefinedTerm", "name": n, "description": d, "inDefinedTermSet": url + "#terms"}
                               for n, d in terms]}


def dataset_ld(url: str, fr: bool) -> dict:
    return {"@context": "https://schema.org", "@type": "Dataset",
            "name": "Paysage des modèles de décision" if fr else "Decision model landscape",
            "description": ("Produits et modèles qui tirent une décision typée d'un texte, propriétaires et ouverts : licence, "
                            "déploiement, sortie, probabilités, prix, source. Relevé le " + TODAY + ".") if fr else
                           ("Products and models that turn a text into a typed decision, proprietary and open: licence, "
                            "deployment, output, probabilities, price, source. Checked on " + TODAY + "."),
            "url": url, "dateModified": TODAY, "creator": {"@type": "Person", "name": "Axel Montzamir"},
            "distribution": [{"@type": "DataDownload", "encodingFormat": "text/csv", "contentUrl": SITE + "/data/landscape.csv"}]}


# ---------------------------------------------------------------------------------------------
# Sidebar, sitemap, llms.txt
# ---------------------------------------------------------------------------------------------
SIDE_EN = {"LEARN": ('<a href="/get-started/"', '<a href="/learn/system-one-models/">System One models</a>'),
           "COMPARE": ('<a href="/benchmarks/"', '<a href="/compare/landscape/">Landscape</a>\n    <a href="/alternatives/jev/">JEV alternatives</a>')}
SIDE_FR = {"APPRENDRE": ('<a href="/fr/demarrer/"', '<a href="/fr/apprendre/modeles-system-one/">Modèles System One</a>'),
           "COMPARER": ('<a href="/fr/resultats/"', '<a href="/fr/comparer/paysage/">Paysage</a>\n    <a href="/fr/alternatives/jev/">Alternatives à JEV</a>')}


def add_side_links(t: str, fr: bool, current: str | None) -> str:
    for kick, (anchor, links) in (SIDE_FR if fr else SIDE_EN).items():
        aside = re.search(r"<aside class=\"side\".*?</aside>", t, re.S)
        if aside and links.split('"')[1] in aside[0]:
            continue
        # insert the new links after the group's first link line
        m = re.search(rf'(<div class="kick">{kick}</div>\n\s*{re.escape(anchor)}[^\n]*\n)', t)
        if m:
            t = t[:m.end()] + "    " + links + "\n" + t[m.end():]
    if current:
        t = t.replace(f'<a href="{current}">', f'<a href="{current}" aria-current="page">', 1)
    return t


def main():
    write_csv()
    en_tpl = (ROOT / "caveats" / "index.html").read_bytes().decode("utf-8")
    fr_tpl = (ROOT / "fr" / "vigilance" / "index.html").read_bytes().decode("utf-8")
    for page in PAGES:
        for fr in (False, True):
            path = page["fr" if fr else "en"]
            url = SITE + path
            if page["key"] == "pillar":
                html, terms = pillar(fr)
                extra = [terms_ld(terms, url, fr)]
            elif page["key"] == "landscape":
                html, extra = landscape(fr), [dataset_ld(url, fr)]
            else:
                html, faq = alternatives(fr)
                extra = [faq_ld(faq)]
            t = build_page(fr_tpl if fr else en_tpl, page, fr, html, extra)
            out = ROOT / path.strip("/") / "index.html"
            out.parent.mkdir(parents=True, exist_ok=True)
            t = add_side_links(t, fr, path)
            out.write_bytes(t.encode("utf-8"))
            print("wrote", out.relative_to(ROOT))
    # every other page: the new links in its sidebar
    for f in sorted(ROOT.rglob("index.html")):
        rel = "/" + f.parent.relative_to(ROOT).as_posix().strip(".") + "/"
        rel = rel.replace("//", "/")
        if any(rel in (pg["en"], pg["fr"]) for pg in PAGES):
            continue
        raw = f.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        t = raw.replace("\r\n", "\n")
        fr = '<html lang="fr"' in t
        new = add_side_links(t, fr, None)
        if new != t:
            f.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("utf-8"))
            print("sidebar", f.relative_to(ROOT))
    # sitemap
    sm = ROOT / "sitemap.xml"
    raw = sm.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    t = raw.replace("\r\n", "\n")
    for page in PAGES:
        for path in (page["en"], page["fr"]):
            if f"<loc>{SITE}{path}</loc>" in t:
                continue
            t = t.replace("</urlset>",
                          f"  <url>\n    <loc>{SITE}{path}</loc>\n"
                          f'    <xhtml:link rel="alternate" hreflang="en" href="{SITE}{page["en"]}"/>\n'
                          f'    <xhtml:link rel="alternate" hreflang="fr" href="{SITE}{page["fr"]}"/>\n'
                          f'    <xhtml:link rel="alternate" hreflang="x-default" href="{SITE}{page["en"]}"/>\n'
                          f"    <lastmod>{TODAY}</lastmod>\n  </url>\n</urlset>")
    sm.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
    # llms.txt
    lt = ROOT / "llms.txt"
    raw = lt.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    t = raw.replace("\r\n", "\n")
    lines = ("- [System One models](https://kahn1.com/learn/system-one-models/): what a System One (decision) model is, how it reads typed answers from logits, the families, how to choose\n"
             "- [Modèles System One (français)](https://kahn1.com/fr/apprendre/modeles-system-one/): la même page en français\n"
             "- [Decision model landscape](https://kahn1.com/compare/landscape/): every proprietary and open way to turn text into a typed decision, with licence, output, price and source (CSV: https://kahn1.com/data/landscape.csv)\n"
             "- [Paysage des modèles de décision (français)](https://kahn1.com/fr/comparer/paysage/): la même page en français\n"
             "- [Open-source alternatives to JEV](https://kahn1.com/alternatives/jev/): Clef, JevK5, Kahn1 and others, JEV-compatible APIs, like-for-like figures, when JEV is still the right pick\n"
             "- [Alternatives open source à JEV (français)](https://kahn1.com/fr/alternatives/jev/): la même page en français\n")
    if "learn/system-one-models" not in t:
        t = t.replace("## Pages\n\n", "## Pages\n\n" + lines, 1)
    lt.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))


if __name__ == "__main__":
    main()
