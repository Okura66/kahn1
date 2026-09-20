"""Construction pipeline for training and evaluation datasets.

Downloads Hugging Face datasets and serializes them to the intermediate JSONL schema:

    {"state": "...", "kind": "choice", "prompt": "...",
     "options": ["...", "..."], "label": 2, "source": "clinc150"}

Training sources (200k+ instances):
  Choice : CLINC150 (including out-of-scope), AG News, DBpedia-14, GoEmotions
  Score  : Yelp Review Full
  Noul   : MNLI, QNLI, BoolQ, ANLI

Reserved evaluation sources:
  Strict invariant: The evaluation split is partitioned at the DATASET LEVEL, not per row.
  Banking77, MASSIVE, and SST-5 are strictly reserved for zero-shot generalization
  benchmarking and never appear in training data mixtures.

Usage:
    python training/build_dataset.py --out data/train.jsonl --eval-out data/eval.jsonl --max-per-source 30000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Reserved Evaluation Datasets (Never seen during training)
# ---------------------------------------------------------------------------

EVAL_CHOICE_SOURCES = {
    "banking77",  # 77 unobserved banking intent classes
    "massive",    # 60 unobserved user intent classes (SetFit/amazon_massive_intent_en-US)
}
EVAL_SCORE_SOURCES = {
    "sst5_eval",  # 5 unobserved sentence polarity levels (SetFit/sst5)
}
EVAL_NOUL_SOURCES: set[str] = set()

# Training dataset registry
TRAIN_SOURCES = {
    "clinc150",
    "ag_news",
    "dbpedia_14",
    "go_emotions",
    "yelp_full",
    "tweet_sentiment",
    "medical_triage",
    "legal_triage",
    "csat_sentiment",
    "mnli",
    "boolq",
    "qnli",
    "anli",
}


# ---------------------------------------------------------------------------
# Primitive conversion helpers
# ---------------------------------------------------------------------------

def _ex_choice(state: str, prompt: str, options: list[str], label: int, source: str) -> dict:
    """Create a typed Choice instance. label = -1 indicates absence of valid options ('other')."""
    return {
        "state": state,
        "kind": "choice",
        "prompt": prompt,
        "options": options,
        "label": label,
        "source": source,
    }


def _ex_score(state: str, prompt: str, levels: list[str], label: int, source: str) -> dict:
    """Create a typed Score instance (invariant natural semantic ordering)."""
    return {
        "state": state,
        "kind": "score",
        "prompt": prompt,
        "levels": levels,
        "label": label,
        "source": source,
    }


def _ex_noul(state: str, statement: str, label: int, source: str) -> dict:
    """Create a typed Noul instance (binary: 1=yes/entailment, 0=no/contradiction/neutral)."""
    return {
        "state": state,
        "kind": "noul",
        "statement": statement,
        "label": label,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Training Loaders (Compatible with datasets >= 5.0 and Parquet repositories)
# ---------------------------------------------------------------------------

def load_clinc150() -> list[dict]:
    """CLINC150 with out-of-scope (oos) split.

    Explicitly teaches the fallback rejection class 'none of the options apply' (allow_other).
    The out-of-scope label "oos" is mapped to label index -1.
    """
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    examples = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for row in ds[split]:
            text = row["text"]
            intent_idx = row["intent"]
            intent_name = names[intent_idx]
            is_oos = (intent_name == "oos")
            examples.append(_ex_choice(
                state=text,
                prompt="Quelle est l'intention de l'utilisateur ?",
                options=[intent_name] if not is_oos else [],
                label=0 if not is_oos else -1,
                source="clinc150",
            ))
    return examples


def load_ag_news() -> list[dict]:
    """AG News: 4 thematic categories (World, Sports, Business, Sci/Tech)."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    examples = []
    for row in ds["train"]:
        examples.append(_ex_choice(
            state=row["text"],
            prompt="Quelle est la catégorie de cet article ?",
            options=labels,
            label=row["label"],
            source="ag_news",
        ))
    return examples


def load_dbpedia() -> list[dict]:
    """DBpedia-14: 14 encyclopedic topic categories."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/dbpedia_14")
    labels = [
        "Company", "EducationalInstitution", "Artist",
        "Athlete", "OfficeHolder", "MeanOfTransportation",
        "Building", "Animal", "Plant", "Album", "Film",
        "WrittenWork", "NaturalPlace", "Village",
    ]
    examples = []
    for row in ds["train"]:
        examples.append(_ex_choice(
            state=row["content"],
            prompt="Quelle est la catégorie de cet article ?",
            options=labels,
            label=row["label"],
            source="dbpedia_14",
        ))
    return examples


def load_go_emotions() -> list[dict]:
    """GoEmotions: 28 fine-grained emotional nuances."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions")
    names = ds["train"].features["labels"].feature.names
    examples = []
    for row in ds["train"]:
        lbls = row["labels"]
        if not lbls:
            continue
        examples.append(_ex_choice(
            state=row["text"],
            prompt="Quelle est l'émotion principale de ce texte ?",
            options=names,
            label=lbls[0],
            source="go_emotions",
        ))
    return examples


def load_yelp_full(anti_midpoint: bool = True, max_total: int = 20000) -> list[dict]:
    """Yelp Review Full: 1 to 5 star customer ratings with anti-midpoint class weighting."""
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full")
    levels = ["1 étoile", "2 étoiles", "3 étoiles", "4 étoiles", "5 étoiles"]

    if anti_midpoint:
        # Quotas favoring polarity extremes and mitigating median midpoint bias (3 stars = 10%)
        quotas = {0: int(max_total * 0.25), 1: int(max_total * 0.20),
                  2: int(max_total * 0.10), 3: int(max_total * 0.20),
                  4: int(max_total * 0.25)}
        counts = {i: 0 for i in range(5)}
        examples = []
        for row in ds["train"]:
            lbl = row["label"]
            if counts[lbl] < quotas.get(lbl, 0):
                examples.append(_ex_score(
                    state=row["text"],
                    prompt="Quelle est la note de cet avis ?",
                    levels=levels,
                    label=lbl,
                    source="yelp_full",
                ))
                counts[lbl] += 1
            if len(examples) >= max_total:
                break
        return examples

    examples = []
    for row in ds["train"]:
        examples.append(_ex_score(
            state=row["text"],
            prompt="Quelle est la note de cet avis ?",
            levels=levels,
            label=row["label"],
            source="yelp_full",
        ))
    return examples


def load_tweet_sentiment(max_per_class: int = 4000) -> list[dict]:
    """Tweet Sentiment Extraction: 3 polarity classes (negative, neutral, positive).

    Applies anti-midpoint weighting: 40% negative, 20% neutral, 40% positive.
    """
    from datasets import load_dataset
    ds = load_dataset("SetFit/tweet_sentiment_extraction")
    levels = ["négatif", "neutre", "positif"]
    # label mapping: 0=negative, 1=neutral, 2=positive
    quotas = {0: max_per_class, 1: max_per_class // 2, 2: max_per_class}
    counts = {0: 0, 1: 0, 2: 0}
    examples = []
    for row in ds["train"]:
        lbl = row["label"]
        if lbl in quotas and counts[lbl] < quotas[lbl]:
            examples.append(_ex_score(
                state=row["text"],
                prompt="Quelle est la polarité de ce tweet ?",
                levels=levels,
                label=lbl,
                source="tweet_sentiment",
            ))
            counts[lbl] += 1
        if all(counts[k] >= quotas[k] for k in quotas):
            break
    return examples


def load_enterprise_medical_triage() -> list[dict]:
    """Emergency medical triage domain: 4 ordinal severity levels."""
    levels = ["Conseil / Automédication", "Consultation Simple", "Urgence Relative", "Urgence Vitale"]
    cases = [
        # Level 0: Self-care / Advice
        ("Patient de 28 ans présentant un rhume léger sans fièvre depuis 24h.", 0),
        ("Légère courbature après séance de sport, pas de douleur articulaire.", 0),
        ("Petite égratignure superficielle sur l'avant-bras après jardinage.", 0),
        ("Demande de renouvellement de traitement contraceptif habituel.", 0),
        ("Fatigue passagère en fin de semaine d'examen, bilan sanguin récent normal.", 0),
        # Level 1: Routine consultation
        ("Fièvre à 38.5°C depuis 48h accompagnée d'une toux grasse et maux de gorge.", 1),
        ("Entorse bénigne de la cheville suite à faux pas, appui encore possible avec boiterie.", 1),
        ("Éruption cutanée prurigineuse apparue hier sur les cuisses, sans gêne respiratoire.", 1),
        ("Douleur lombaire modérée survenue après port de charge lourde sans irradiation.", 1),
        ("Otite douloureuse unilatérale chez l'adulte sans écoulement purulent.", 1),
        # Level 2: Relative emergency
        ("Coupure profonde à la main avec saignement continu malgré compression de 10 min.", 2),
        ("Crise d'asthme inhabituelle ne cédant que partiellement à 2 bouffées de salbutamol.", 2),
        ("Douleur abdominale vive en fosse iliaque droite fébrile suspectant appendicite.", 2),
        ("Nourrisson de 3 mois avec fièvre à 39.2°C, léthargique et refusant le biberon.", 2),
        ("Traumatisme crânien avec brève perte de connaissance de 30 secondes chez un footballeur.", 2),
        # Level 3: Vital emergency
        ("Homme de 58 ans avec douleur thoracique constrictive irradiant dans le bras gauche et sueurs.", 3),
        ("Choc anaphylactique avec œdème de Quincke, stridor et cyanose après piqûre de guêpe.", 3),
        ("Déficit moteur brutal hémi-corporel droit avec aphasie brutale depuis 15 minutes (AVC).", 3),
        ("Arrêt cardio-respiratoire chez un adulte inconscient en arrêt de ventilation spontanée.", 3),
        ("Polytraumatisé inconscient incarcéré après accident de la route à haute cinétique.", 3),
    ]
    examples = []
    prompts = [
        "Quel est le niveau de gravité et d'urgence de cette situation ?",
        "Évaluez le degré de priorité clinique de ce patient.",
    ]
    for p in prompts:
        for text, lbl in cases:
            examples.append(_ex_score(
                state=text,
                prompt=p,
                levels=levels,
                label=lbl,
                source="medical_triage",
            ))
    return examples


def load_enterprise_legal_triage() -> list[dict]:
    """Legal contract clause classification (6 typed option categories)."""
    options = [
        "Confidentialité",
        "Résiliation",
        "Responsabilité & Indemnisation",
        "Propriété Intellectuelle",
        "Force Majeure",
        "Loi Applicable & Juridiction",
    ]
    clauses = [
        # 0: Confidentiality
        ("Chaque partie s'engage à préserver le secret le plus absolu sur les données confidentielles transmises.", 0),
        ("The recipient agrees to hold and maintain all proprietary information in strict confidence.", 0),
        ("Les obligations de non-divulgation survivront à l'extinction du contrat pour une durée de 5 ans.", 0),
        # 1: Termination
        ("Le présent accord peut être résilié par l'une ou l'autre des parties avec un préavis écrit de 30 jours.", 1),
        ("Either party may immediately terminate this agreement upon material breach not cured within 15 days.", 1),
        ("En cas de liquidation judiciaire du prestataire, le client pourra prononcer la rupture de plein droit.", 1),
        # 2: Liability
        ("La responsabilité totale de chaque partie sera expressément plafonnée au montant des sommes versées.", 2),
        ("In no event shall either party be liable for any indirect, consequential or punitive damages.", 2),
        ("Le prestataire indemnisera le client contre tout recours de tiers résultant d'une négligence grave.", 2),
        # 3: Intellectual Property
        ("Tous les droits d'auteur, brevets et codes sources développés demeurent la propriété exclusive du client.", 3),
        ("All Intellectual Property rights arising out of the performance of the services shall belong to Licensor.", 3),
        ("La présente concession de licence est non-exclusive, mondiale et incessible pour la durée du projet.", 3),
        # 4: Force Majeure
        ("Aucune des parties ne sera tenue pour responsable d'un manquement causé par une catastrophe naturelle ou guerre.", 4),
        ("Neither party shall be liable for delay caused by acts of God, pandemic, or government restrictions.", 4),
        ("L'évènement imprévisible et irrésistible suspend les obligations contractuelles pendant sa durée.", 4),
        # 5: Governing Law
        ("Le présent contrat est régi et interprété selon le droit français, compétence exclusive aux tribunaux de Paris.", 5),
        ("This agreement shall be governed by and construed in accordance with the laws of the State of Delaware.", 5),
        ("Tout litige sera soumis à l'arbitrage de la Chambre de Commerce Internationale.", 5),
    ]
    examples = []
    prompts = [
        "Quelle est la catégorie juridique de cette clause contractuelle ?",
        "Which legal category best describes this contractual clause?",
    ]
    for p in prompts:
        for text, lbl in clauses:
            examples.append(_ex_choice(
                state=text,
                prompt=p,
                options=options,
                label=lbl,
                source="legal_triage",
            ))
    return examples


def load_customer_satisfaction_csat() -> list[dict]:
    """E-commerce and SaaS customer support reviews (5 ordinal CSAT satisfaction levels)."""
    levels = ["Très insatisfait", "Insatisfait", "Neutre", "Satisfait", "Très satisfait"]
    tickets = [
        ("Produit reçu brisé en mille morceaux, service client injoignable, je demande un remboursement immédiat !", 0),
        ("Inadmissible : prélèvement indu sur mon compte bancaire et aucune réponse depuis 3 semaines.", 0),
        ("Délai de livraison non respecté de plus de 10 jours, l'emballage était déchiré.", 1),
        ("L'application plante fréquemment lors du paiement, expérience frustrante à améliorer.", 1),
        ("Commande reçue dans les temps. Produit conforme à la description, ni plus ni moins.", 2),
        ("Article standard qui fait le travail, rapport qualité prix correct sans enthousiasme.", 2),
        ("Bonne expérience d'achat globale, livraison rapide et produit conforme à mes attentes.", 3),
        ("Service après-vente courtois qui a résolu mon problème de mot de passe en 10 minutes.", 3),
        ("Absolument parfait ! Qualité exceptionnelle, expédié en 24h et emballage soigné. Je recommande vivement !", 4),
        ("Meilleur support client jamais testé, réponse ultra rapide et geste commercial généreux, bravo !", 4),
    ]
    examples = []
    prompts = [
        "Quel est le niveau de satisfaction exprimé dans cet avis ?",
        "Rate the customer satisfaction level of this message.",
    ]
    for p in prompts:
        for text, lbl in tickets:
            examples.append(_ex_score(
                state=text,
                prompt=p,
                levels=levels,
                label=lbl,
                source="csat_sentiment",
            ))
    return examples


def load_mnli() -> list[dict]:
    """MNLI: premise/hypothesis textual inference (entailment = 1, other = 0)."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "mnli")
    examples = []
    for row in ds["train"]:
        examples.append(_ex_noul(
            state=row["premise"],
            statement=row["hypothesis"],
            label=1 if row["label"] == 0 else 0,
            source="mnli",
        ))
    return examples


def load_boolq() -> list[dict]:
    """BoolQ: boolean yes/no reading comprehension questions over text passages."""
    from datasets import load_dataset
    ds = load_dataset("google/boolq")
    examples = []
    for row in ds["train"]:
        examples.append(_ex_noul(
            state=row["passage"],
            statement=row["question"],
            label=1 if row["answer"] else 0,
            source="boolq",
        ))
    return examples


def load_qnli() -> list[dict]:
    """QNLI: question/context sentence entailment (0=entailment, 1=not_entailment)."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "qnli")
    examples = []
    for row in ds["train"]:
        examples.append(_ex_noul(
            state=row["sentence"],
            statement=row["question"],
            label=1 if row["label"] == 0 else 0,
            source="qnli",
        ))
    return examples


def load_anli() -> list[dict]:
    """ANLI: Adversarial NLI benchmark rounds (Rounds 1, 2, 3)."""
    from datasets import load_dataset
    ds = load_dataset("facebook/anli")
    examples = []
    for split in ["train_r1", "train_r2", "train_r3"]:
        if split in ds:
            for row in ds[split]:
                examples.append(_ex_noul(
                    state=row["premise"],
                    statement=row["hypothesis"],
                    label=1 if row["label"] == 0 else 0,
                    source="anli",
                ))
    return examples


# ---------------------------------------------------------------------------
# Reserved Evaluation Loaders (Never seen during training)
# ---------------------------------------------------------------------------

def load_banking77_eval() -> list[dict]:
    """Banking77 reserved for evaluation (77 unobserved banking intent categories)."""
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77")
    names = sorted(list(set(ds["train"]["label_text"])))
    name_to_idx = {name: i for i, name in enumerate(names)}
    examples = []
    for row in ds["test"]:
        lbl_text = row["label_text"]
        examples.append(_ex_choice(
            state=row["text"],
            prompt="Quelle est la catégorie bancaire de cette demande ?",
            options=names,
            label=name_to_idx[lbl_text],
            source="banking77",
        ))
    return examples


def load_massive_eval() -> list[dict]:
    """MASSIVE reserved for evaluation (60 unobserved user intent categories)."""
    from datasets import load_dataset
    ds = load_dataset("SetFit/amazon_massive_intent_en-US")
    names = sorted(list(set(ds["train"]["label_text"])))
    name_to_idx = {name: i for i, name in enumerate(names)}
    examples = []
    for row in ds["test"]:
        lbl_text = row["label_text"]
        examples.append(_ex_choice(
            state=row["text"],
            prompt="What is the user's intent?",
            options=names,
            label=name_to_idx[lbl_text],
            source="massive",
        ))
    return examples


def load_sst5_eval() -> list[dict]:
    """SST-5 reserved for evaluation (5 unobserved fine-grained sentence polarity levels)."""
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5")
    levels = ["très négatif", "négatif", "neutre", "positif", "très positif"]
    examples = []
    for row in ds["test"]:
        examples.append(_ex_score(
            state=row["text"],
            prompt="Quelle est la polarité de cette phrase ?",
            levels=levels,
            label=row["label"],
            source="sst5_eval",
        ))
    return examples


# ---------------------------------------------------------------------------
# Unified Construction Pipeline
# ---------------------------------------------------------------------------

def build(
    out: str,
    eval_out: str,
    max_per_source: int | None = 30000,
    seed: int = 42,
) -> tuple[int, int]:
    """Build train.jsonl and eval.jsonl dataset files with strict source isolation."""
    rng = random.Random(seed)
    train: list[dict] = []
    eval_: list[dict] = []

    # Theoretical isolation verification
    all_eval_sources = EVAL_CHOICE_SOURCES | EVAL_SCORE_SOURCES | EVAL_NOUL_SOURCES
    overlap = TRAIN_SOURCES.intersection(all_eval_sources)
    if overlap:
        raise ValueError(f"DATASET LEAK VIOLATION: datasets present in both train and eval registries: {overlap}")

    train_loaders = {
        "clinc150": load_clinc150,
        "ag_news": load_ag_news,
        "dbpedia_14": load_dbpedia,
        "go_emotions": load_go_emotions,
        "yelp_full": load_yelp_full,
        "tweet_sentiment": load_tweet_sentiment,
        "medical_triage": load_enterprise_medical_triage,
        "legal_triage": load_enterprise_legal_triage,
        "csat_sentiment": load_customer_satisfaction_csat,
        "mnli": load_mnli,
        "boolq": load_boolq,
        "qnli": load_qnli,
        "anli": load_anli,
    }

    eval_loaders = {
        "banking77": load_banking77_eval,
        "massive": load_massive_eval,
        "sst5_eval": load_sst5_eval,
    }

    print("[build] Loading training sources...")
    for name, fn in train_loaders.items():
        try:
            exs = fn()
            if max_per_source and len(exs) > max_per_source:
                rng.shuffle(exs)
                exs = exs[:max_per_source]
            train.extend(exs)
            print(f"  + TRAIN : {name} -> {len(exs)} instances")
        except Exception as e:
            print(f"  ! TRAIN failure for {name}: {e}")

    print("\n[build] Loading reserved evaluation sources...")
    for name, fn in eval_loaders.items():
        try:
            exs = fn()
            eval_.extend(exs)
            print(f"  + EVAL RESERVED : {name} -> {len(exs)} instances")
        except Exception as e:
            print(f"  ! EVAL failure for {name}: {e}")

    # Strict empirical leak check
    train_srcs = {ex["source"] for ex in train}
    eval_srcs = {ex["source"] for ex in eval_}
    leak = train_srcs.intersection(eval_srcs)
    if leak:
        raise RuntimeError(f"DATASET LEAK DETECTED BETWEEN TRAIN AND EVAL: {leak}")

    # Shuffle training instances to break consecutive blocks
    rng.shuffle(train)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(eval_out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_out, "w", encoding="utf-8") as f:
        for ex in eval_:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n=======================================================")
    print(f"Datasets generated successfully:")
    print(f"  TRAIN : {len(train):,} instances written to {out}")
    print(f"  EVAL  : {len(eval_):,} instances written to {eval_out}")
    print(f"  Strict isolation guaranteed: 0 source overlap")
    print(f"=======================================================")

    return len(train), len(eval_)


def main():
    ap = argparse.ArgumentParser(description="Construct train and evaluation dataset mixtures for model training.")
    ap.add_argument("--out", default="data/train.jsonl", help="Train dataset output path")
    ap.add_argument("--eval-out", default="data/eval_reserved.jsonl", help="Reserved evaluation dataset output path")
    ap.add_argument("--max-per-source", type=int, default=30000,
                    help="Maximum instances per data source to balance classes")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    build(
        out=args.out,
        eval_out=args.eval_out,
        max_per_source=args.max_per_source,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
