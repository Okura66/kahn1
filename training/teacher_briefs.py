"""Briefs for the teacher author agents (hard decision questions, EN and FR).

Each author agent reads one brief and writes 30 questions (10 documents x 3). The family mix
leans on what the pilot showed small models miss most (temporal_numeric, routing, tradeoff,
multi_hop, judge). Solver agents later answer each question blind; scripts/teacher_agreement.py
keeps a question when both give the author's label.

    python training/teacher_briefs.py data/teacher/t1      # writes t1/briefs/author_NN.md
    python training/teacher_briefs.py data/teacher/t2      # round 2: harder families, yes/no questions
    python training/teacher_briefs.py data/teacher/t3      # round 3: 68 authors in two waves, each
                                                           # with its own names and a per-document plan

Round 3 fights duplicates at the source (training/teacher_catalog.py): every author gets a
document-by-document plan (family, domain, deciding mechanism) and its own people and
organisation names; scripts/teacher_dedup.py then filters what still repeats before the solvers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.teacher_catalog import assign_mechanisms, author_names  # noqa: E402

FAMILIES = {
    "long_policy": "a long policy or terms document with numbered sections, definitions and exceptions; the question "
                   "asks whether an action is permitted or which outcome applies, and the deciding clause is an "
                   "exception, a definition or a later section",
    "tradeoff": "several rules could apply and the document states their precedence; the answer is the action of the "
                "highest-ranked applicable rule",
    "routing": "a ticket or request must go to one of several handlers whose descriptions overlap; one detail decides "
               "the best fit",
    "temporal_numeric": "dates, deadlines, business days, month ends, time zones, durations, amounts, thresholds, "
                        "currency conversion or aggregation; answers need exact computations a quick reading gets wrong",
    "extraction": "final values such as dates, amounts, owners, statuses extracted from a messy email thread, chat or "
                  "log with proposals, corrections, cancellations and reversals",
    "multi_hop": "answers need 2 to 4 facts from different parts of the document combined: lookup tables, rate cards, "
                 "org charts, vendor lists, mappings",
    "judge": "the document contains a request and a response to it (a customer's question and an agent's reply, a task "
             "and a colleague's answer, a spec and an implementation summary); questions decide whether the response "
             "is correct, complete and follows every constraint; errors are subtle: an arithmetic slip, one violated "
             "constraint, an unsupported claim, a missing required part",
    "rubric": "the document is rated on ordinal rubrics of 3 to 5 levels whose descriptions are precise; the right "
              "level depends on details that rule out the neighbouring levels",
    "ambiguous": "the evidence is incomplete or conflicting on some points, so for those the careful answer is an "
                 "explicit \"cannot be determined\" or \"ask for clarification\" option, while other points are settled "
                 "despite looking vague",
    "trap": "surface cues point to wrong answers: a confident note, a headline, a customer's claim, a superseded "
            "version, a similar-looking name, and careful reading gives the right ones",
    "adversarial": "the document contains text that tries to steer the decision (an instruction addressed to the "
                   "classifier or an AI system, a fake \"system note\", a line claiming the answer); correct answers "
                   "ignore it and follow the actual evidence, and the injected text is never the source of the answer",
}

TRIOS = [
    ("long_policy", "tradeoff", "routing"),
    ("temporal_numeric", "extraction", "multi_hop"),
    ("judge", "rubric", "ambiguous"),
    ("trap", "adversarial", "long_policy"),
    ("temporal_numeric", "tradeoff", "routing"),
    ("multi_hop", "judge", "temporal_numeric"),
    ("routing", "ambiguous", "extraction"),
    ("tradeoff", "multi_hop", "adversarial"),
    ("rubric", "trap", "judge"),
    ("temporal_numeric", "long_policy", "routing"),
]

DOMAINS = {
    "en": ["insurance claims", "HR leave and benefits", "e-commerce returns and warranties", "SaaS subscription billing",
           "airline passenger policies", "logistics and shipping", "payroll and expenses", "procurement and invoices",
           "travel bookings", "IT change management", "customer support replies", "software code review",
           "healthcare administration (scheduling, coverage, not clinical advice)", "education and grading",
           "marketing claims review", "retail banking and payments compliance", "telecom contracts",
           "cybersecurity access requests", "property management and maintenance", "grant and scholarship applications",
           "energy utility billing", "car rental and fleet management", "construction subcontracting",
           "data privacy requests (GDPR / CCPA)", "university admissions", "hotel reservations",
           "manufacturing quality control", "public transport fares and refunds", "nonprofit volunteer management",
           "software incident reports"],
    "fr": ["droit du travail (congés, heures supplémentaires, arrêts)", "baux d'habitation et état des lieux",
           "copropriété et assemblées générales", "notariat et successions", "marchés publics et appels d'offres",
           "URSSAF et cotisations sociales", "assurance habitation et automobile", "mutuelle et complémentaire santé",
           "facturation, avoirs et TVA", "e-commerce, retours et garanties légales", "transport ferroviaire de voyageurs",
           "banque et conformité (LCB-FT, virements)", "demandes RGPD (accès, effacement)", "scolarité, examens et notes",
           "support client télécom et box internet", "gestion de sinistres", "recrutement et période d'essai",
           "notes de frais et déplacements professionnels", "logistique et transport routier",
           "maintenance immobilière et syndic", "subventions et appels à projets", "revue de code logiciel",
           "gestion d'incidents informatiques", "restauration collective et hygiène", "agence de voyages et forfaits",
           "formation professionnelle et CPF", "location de véhicules", "crèche et périscolaire",
           "abonnements SaaS et résiliation", "comptabilité fournisseurs et relances"],
}

EXAMPLES = {
    "en": ('"The customer is entitled to a full refund."', '"Cannot be determined from the document"'),
    "fr": ('"Le client a droit à un remboursement intégral."', '"Impossible à déterminer à partir du document"'),
}


def brief(n: int, lang: str, trio: tuple[str, ...], domains: list[str], out_path: Path, prefix: str = "t",
          question_noul: bool = False, mechanisms: list[str] | None = None,
          names: dict[str, list[str]] | None = None, length_rule: int = 0) -> str:
    fam = ", ".join(f'"{f}" ({FAMILIES[f]})' for f in trio)
    if mechanisms:
        # Document k: the family, domain and deciding detail it is built around.
        docs = "\n".join(f"  {k + 1}. family \"{trio[k % len(trio)]}\", domain \"{domains[k // 2]}\": built around this "
                         f"deciding detail: {m}." for k, m in enumerate(mechanisms))
        plan = (f"- Families: {fam}.\n- Follow this plan, one line per document (the 3 questions of a document may "
                f"use other details too, but the first one hinges on the listed one):\n{docs}")
    else:
        plan = (f"- Families (spread evenly over the 10 documents): {fam}.\n"
                f"- Domains (use each one exactly twice): {'; '.join(domains)}.")
    # Wave 1 of round 3 made the correct option the longest one 35 % of the time (chance 23 %):
    # the deciding nuance went into the right answer only. Authors 35-44 got rule 1, and the first
    # of them overcorrected (the correct option never the longest, a cue too); from author 45, rule 2.
    length_text = {
        0: "",
        1: (" Write every wrong option with the same length and level of detail as the correct one "
            "(conditions, amounts, qualifiers): across your choice questions the correct option must be the "
            "longest one no more often than any other option, and never the only one that states a "
            "condition or a nuance."),
        2: (" Write every wrong option with the same length and level of detail as the correct one "
            "(conditions, amounts, qualifiers), so that length says nothing about the answer: across your 13 "
            "choice questions, make the correct option the longest one in about 3 of them (roughly what chance "
            "gives), never in most of them and not in none; and it is never the only option that states a "
            "condition or a nuance."),
    }[length_rule]
    names_rule = ("" if not names else
                  "\n- Names: for people use ONLY names from this list: " + ", ".join(names["people"])
                  + ". For organisations use ONLY these: " + ", ".join(names["organisations"])
                  + ". You may add a department, a product or a place name, but invent no other person or "
                    "company name, and do not reuse one person in more than two documents.")
    noul_ex, cbd = EXAMPLES[lang]
    language = ("Write every document, question, option and level in English." if lang == "en" else
                "Write every document, question, option and level in natural, idiomatic French (French business "
                "and legal context: euros, French dates, French institutions where relevant). Only the JSON keys "
                "and the rationale may be in English.")
    noul_rule = (f'''- "noul": 9 items, label 1 = yes / true, 0 = no / false, balanced (4 or 5 of each). Write 5 of them as a yes/no QUESTION a reviewer would ask about the document (e.g. "Is the requested refund permitted under the policy?", "Does the reply follow every constraint the customer set?"), with "form": "yesno_question", and 4 as a declarative statement to judge true or false ({noul_ex}), with "form": "statement".'''
                 if question_noul else
                 f'- "noul": a single declarative statement to judge true or false against the document ({noul_ex}). Label 1 = true, 0 = false. Balance true and false (4 or 5 of each).')
    return f"""You are writing training data for a small "typed decision" model: it reads a document (the "state") and answers one typed question about it in a single pass, without reasoning out loud. Your job is to write HARD, realistic decision questions whose correct answer depends on a detail that a quick reading gets wrong. Two other models will later solve each question independently; questions they do not both solve are discarded, so every question must have exactly one defensible answer from the document alone.

WRITE 10 documents, and 3 questions per document (30 questions total).
{plan}
- Language: {language}

Document rules:
- Realistic business text with plausible names, numbers, section numbers and dates. No placeholders like [Company].{names_rule}
- Length: 400 to 2000 words. At least 4 of your 10 documents must be over 1000 words, and at least 2 over 1500 words; in long documents put a deciding detail far from the start.
- Each of the 3 questions about a document hinges on a DIFFERENT easy-to-miss detail (an exception, a definition, a threshold, a later amendment, an overriding clause, a correction later in a thread, a time zone).
- Do the arithmetic and date computations exactly (use Python to check them).

Question rules (mix across your 30 questions: about 13 "choice", 9 "noul", 8 "score"):
- "choice": a clear question and 3 to 6 options. Options are full descriptions of outcomes or handlers, written neutrally: no option may give away the answer by wording ("correct", "actually", "properly", "trap", "naive"...), and wrong options must be what a careless or manipulated reader would pick. You may include an option like {cbd} only when that is truly the answer, and it must also appear, as a wrong option, in some questions where the answer is settled. Spread the position of the correct option across first, middle and last.{length_text}
{noul_rule}
- "score": an ordinal scale of 3 to 5 levels, ordered from lowest to highest, each level precisely described (e.g. severity, eligibility tier, compliance, completeness), so that specific details rule the right level in and the neighbouring levels out. Label = 0-based index of the correct level. Spread the correct level over the whole scale: among your 8 score questions, at least 2 must be answered by the LOWEST level and at least 2 by the HIGHEST level.
- The answer must follow from the document alone, without outside knowledge.

OUTPUT: write the 30 questions as JSON Lines (one JSON object per line, UTF-8) to the file
{out_path}
with exactly these fields:
{{"id": "{prefix}{n:02d}-d01-q1", "family": "{trio[0]}", "domain": "{domains[0]}", "lang": "{lang}", "state": "<the full document text>", "kind": "choice", "prompt": "<question>", "options": ["...", "..."], "label": 2, "rationale": "<one or two sentences in English: which detail decides and why the tempting answer is wrong>"}}
For "noul" use "statement" instead of "prompt"/"options" (and label 1/0){' and add the "form" field' if question_noul else ''}. For "score" use "prompt" and "levels" (ordered lowest to highest) instead of "options". Repeat the full document text in "state" for each of its 3 questions. Ids: {prefix}{n:02d}-dNN-qK.

Write the file with a Python script (json.dumps with ensure_ascii=False), not by hand-escaping JSON. When done, validate it with Python (every line parses, required fields present, labels in range, 30 lines, the score-level and noul balance above) and fix any problem. Finish with a one-paragraph summary: counts per family, kind, label position, document lengths, and any question you are unsure about. Do not read or search for any benchmark or other files in the repository; write the content from scratch.
"""


# Round 2 leans on what the 4B still misses on JevBench's hard tier: date and amount
# computation, judging a response, rule precedence, multi-hop lookups.
TRIOS_T2 = [
    ("temporal_numeric", "judge", "tradeoff"),
    ("temporal_numeric", "multi_hop", "routing"),
    ("judge", "tradeoff", "temporal_numeric"),
    ("multi_hop", "temporal_numeric", "judge"),
    ("tradeoff", "routing", "temporal_numeric"),
    ("judge", "ambiguous", "multi_hop"),
    ("temporal_numeric", "extraction", "tradeoff"),
    ("routing", "judge", "long_policy"),
    ("temporal_numeric", "adversarial", "multi_hop"),
    ("tradeoff", "trap", "judge"),
]
# Round 3 leans on what Kahn1 4B v2 still misses most: long policies, judging a response, dates
# and amounts, multi-hop lookups and ambiguity; every other family keeps at least one trio.
TRIOS_T3 = [
    ("long_policy", "judge", "temporal_numeric"),
    ("long_policy", "multi_hop", "ambiguous"),
    ("judge", "temporal_numeric", "multi_hop"),
    ("long_policy", "temporal_numeric", "trap"),
    ("judge", "ambiguous", "long_policy"),
    ("multi_hop", "long_policy", "routing"),
    ("temporal_numeric", "judge", "extraction"),
    ("ambiguous", "multi_hop", "tradeoff"),
    ("long_policy", "rubric", "adversarial"),
    ("judge", "multi_hop", "temporal_numeric"),
]
ROUNDS = {
    "t1": dict(trios=TRIOS, prefix="t", offset=0, question_noul=False),
    "t2": dict(trios=TRIOS_T2, prefix="u", offset=2, question_noul=True),
    "t3": dict(trios=TRIOS_T3, prefix="w", offset=4, question_noul=True, catalog=True),
}
EXTRA_DOMAINS = {
    "en": ["clinical trial site contracts (administrative terms only)", "airport ground handling", "freight customs brokerage",
           "franchise agreements", "event venue hire", "pension plan administration", "cloud infrastructure SLAs",
           "academic research grants", "municipal permits and licensing", "veterinary clinic billing",
           "co-working memberships", "warehouse safety audits", "film and media licensing", "esports league rules",
           "agricultural cooperatives"],
    "fr": ["fiscalité des particuliers (impôt sur le revenu)", "permis de construire et urbanisme",
           "contrats de maintenance industrielle", "aides agricoles (PAC)", "auto-entrepreneurs et micro-entreprises",
           "location saisonnière", "associations loi 1901", "transport aérien et indemnisation",
           "énergie (contrats d'électricité et de gaz)", "pompes funèbres et obsèques"],
}


def _existing_texts(root: Path) -> list[str]:
    """Documents of the earlier rounds and of JevBench, whose names a new round must not reuse."""
    import json

    teacher = root.parent
    files = [f for d in ("pilot", "t1", "t2") for f in sorted((teacher / d).glob("author_*.jsonl"))]
    files.append(teacher.parent / "jevbench_eval.jsonl")
    texts = set()
    for f in files:
        if f.exists():
            texts.update(json.loads(l)["state"] for l in f.open(encoding="utf-8") if l.strip())
    return sorted(texts)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/teacher/t1")
    rd = ROUNDS[root.name]
    (root / "briefs").mkdir(parents=True, exist_ok=True)
    # About 70 % English (JevBench, the target, is English), 30 % French so the model does not
    # drift there. English repeats the trios holding the weakest pilot families; the French
    # trios still cover every family.
    plan = [("en", t) for t in range(10)] + [("en", t) for t in (4, 5, 7, 9)] + \
           [("fr", t) for t in (0, 1, 2, 3, 5, 8)]
    if rd.get("catalog"):
        # 68 authors, two waves of 34 (authors 1-34, then 35-68), each wave 24 English and 10 French.
        wave = [("en", t % 10) for t in range(24)] + [("fr", (t * 3) % 10) for t in range(10)]
        plan = wave + [(lang, (t + 5) % 10) for lang, t in wave]
    domains_of = {lang: DOMAINS[lang] + (EXTRA_DOMAINS[lang] if rd.get("catalog") else []) for lang in DOMAINS}
    trios = [rd["trios"][ti] for _, ti in plan]
    mech = assign_mechanisms(trios) if rd.get("catalog") else [None] * len(plan)
    names = (author_names(len(plan), [lang for lang, _ in plan], _existing_texts(root)) if rd.get("catalog")
             else [None] * len(plan))
    seen = {"en": 0, "fr": 0}
    for n, (lang, ti) in enumerate(plan, 1):
        pool, j = domains_of[lang], seen[lang]
        seen[lang] += 1
        # 5 domains per author, rotated so each language's pool is covered and neighbours differ.
        domains = [pool[(j * 5 + k + rd["offset"]) % len(pool)] for k in range(5)]
        out = (root / f"author_{n:02d}.jsonl").resolve()
        (root / "briefs" / f"author_{n:02d}.md").write_text(
            brief(n, lang, rd["trios"][ti], domains, out, rd["prefix"], rd["question_noul"],
                  mech[n - 1], names[n - 1],
                  length_rule=(0 if not rd.get("catalog") or n <= 34 else 1 if n <= 44 else 2)),
            encoding="utf-8")
    print(f"{len(plan)} briefs in {root / 'briefs'}")


if __name__ == "__main__":
    main()
