"""Teacher questions -> a hard dev split and the v5 training mixture.

- Verified = both blind solvers gave the author's label (scripts/teacher_agreement.py). A
  question a solver contradicted is dropped everywhere; unverified questions (no solver pass,
  about 1 % label noise going by the checked ones) are used for training only.
- dev: the pilot's kept questions plus whole documents drawn from the verified t1 ones,
  English and French in proportion, so no document is on both sides.
- train: the other t1 questions, repeated --teacher-repeat times (every pass permutes the
  options), mixed with a subsample of data/train_v4.jsonl that cuts short classification.

    python training/build_teacher_mix.py --dev-questions 200 --teacher-repeat 4     # data/train_v5.jsonl

The second 4B mixture adds the round-2 teacher questions (data/teacher/t2, all for training:
the dev split stays the same, so scores stay comparable), yes/no questions from BoolQ,
more CLINC150 intents from data/train.jsonl, and keeps the sentiment scales whole:

    python training/build_teacher_mix.py --round2 --out data/train_4b_v2.jsonl

The run of 2026-09-30 (a ~20 % teacher share):

    python training/build_teacher_mix.py --round2 --trim --teacher-repeat 5 --round2-repeat 4
        --replay arc=1000,csqa=1000,mmlu_pro_val=70 --out data/train_4b_v2.jsonl

--replay adds multiple-choice reasoning questions the base model already answers (ARC,
CommonsenseQA, the 70 MMLU-Pro validation questions; never MMLU-Pro test), so training on
our tasks erodes less of what it knows (JevK5 v0.2 replays such sets): --replay default, or
--replay arc=1500,csqa=1500,mmlu_pro_val=70 (0 or --replay none turns a source off).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "data" / "teacher"

# Sources of data/train_v4.jsonl and how many rows each keeps (None = all).
# Round 2: the sentiment scales whole (Score lost 2.8 points on the held-out set with the cut),
# more intents (short intent classification lost ground on JevBench) and yes/no questions
# (JevBench asks its judgment calls as questions), the last two from data/train.jsonl.
V4_KEEP_R2 = {"sharc": 2500, "wanli": 2500, "docnli": 2000, "quality": None, "clinc150": None,
              "amazon_reviews": None, "yelp_full": None, "go_emotions": 1000, "ag_news": 800, "dbpedia_14": 800,
              "imdb": 700, "tweet_sentiment": None}
EXTRA_R2 = {"clinc150": 2500, "boolq": 3000}
# --trim: a larger teacher share (~20 % with --teacher-repeat 5 --round2-repeat 4) by halving
# the long-document NLI and the extra yes/no and intent rows; the sentiment scales stay whole.
TRIM_R2 = {"sharc": 1250, "wanli": 1250, "docnli": 1000, "go_emotions": 600, "ag_news": 500, "dbpedia_14": 500}
EXTRA_TRIM_R2 = {"clinc150": 1000, "boolq": 1500}
V4_KEEP = {"sharc": 2500, "wanli": 2500, "docnli": 2000, "quality": None, "clinc150": None,
           "amazon_reviews": 700, "yelp_full": 700, "go_emotions": 600, "ag_news": 500, "dbpedia_14": 500,
           "imdb": 500, "tweet_sentiment": 400}
KEYS = ("state", "kind", "prompt", "options", "levels", "statement", "label", "form")

REPLAY_DEFAULT = {"arc": 1500, "csqa": 1500, "mmlu_pro_val": 70}
REPLAY_PROMPT = "Which option answers the question correctly?"


def load_replay(counts: dict[str, int], rng: random.Random) -> list[dict]:
    """Train splits of ARC (Challenge first, then Easy) and CommonsenseQA, MMLU-Pro validation.

    The question is the state and the options stay its own (fixed_options): augmentation
    permutes them but never swaps in distractors from other sources.
    """
    from datasets import load_dataset

    def lettered(r):
        c = r["choices"]
        return c["text"], c["label"].index(r["answerKey"])

    pools = {}
    if counts.get("arc"):
        arc = [("arc", r) for cfg in ("ARC-Challenge", "ARC-Easy")
               for r in load_dataset("allenai/ai2_arc", cfg)["train"]]
        pools["arc"] = [(s, r["question"], *lettered(r)) for s, r in arc]
    if counts.get("csqa"):
        pools["csqa"] = [("csqa", r["question"], *lettered(r))
                         for r in load_dataset("tau/commonsense_qa")["train"]]
    if counts.get("mmlu_pro_val"):
        pools["mmlu_pro_val"] = [("mmlu_pro_val", r["question"], r["options"], r["answer_index"])
                                 for r in load_dataset("TIGER-Lab/MMLU-Pro")["validation"]]
    out = []
    for name, rows in pools.items():
        if name != "arc":          # ARC keeps Challenge first; the others are drawn at random
            rng.shuffle(rows)
        # A question listing one answer twice (a few CommonsenseQA ones) fails ChoiceQuestion.
        rows = [r for r in rows if len(set(r[2])) == len(r[2])]
        for src, question, options, label in rows[:counts[name]]:
            out.append({"state": question.strip(), "kind": "choice", "prompt": REPLAY_PROMPT,
                        "options": list(options), "label": label, "source": f"replay-{src}",
                        "fixed_options": True})
    return out


def parse_replay(spec: str | None) -> dict[str, int]:
    if spec in (None, "none", "off"):
        return {}
    if spec == "default":
        return dict(REPLAY_DEFAULT)
    counts = {k: int(v) for k, v in (part.split("=") for part in spec.split(","))}
    unknown = set(counts) - set(REPLAY_DEFAULT)
    if unknown:
        raise SystemExit(f"unknown replay source(s) {sorted(unknown)}; expected {sorted(REPLAY_DEFAULT)}")
    return {k: v for k, v in counts.items() if v > 0}


def load(pattern: str, d: Path) -> dict[str, dict]:
    return {r["id"]: r for f in sorted(d.glob(pattern)) for r in map(json.loads, f.open(encoding="utf-8"))}


def row(r: dict) -> dict:
    out = {k: r[k] for k in KEYS if k in r}
    return {**out, "source": "teacher-" + r["family"], "lang": r["lang"], "fixed_options": True, "id": r["id"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev-questions", type=int, default=200)
    ap.add_argument("--teacher-repeat", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round2", action="store_true", help="add data/teacher/t2, BoolQ, more CLINC150")
    ap.add_argument("--round2-repeat", type=int, default=3)
    ap.add_argument("--trim", action="store_true", help="with --round2: fewer NLI, yes/no and intent rows")
    ap.add_argument("--replay", default=None, metavar="SPEC",
                    help='"default" (%s), "src=n,..." or "none"' % ",".join(f"{k}={v}" for k, v in REPLAY_DEFAULT.items()))
    ap.add_argument("--out", default="data/train_v5.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    t1 = T / "t1"
    authors, A, B = load("author_*.jsonl", t1), load("answers_A_*.jsonl", t1), load("answers_B_*.jsonl", t1)
    status = {}
    for qid, r in authors.items():
        a, b = A.get(qid, {}).get("answer"), B.get(qid, {}).get("answer")
        if a is None and b is None:
            status[qid] = "unverified"
        elif a == b == r["label"]:
            status[qid] = "verified"
        elif (a is None or a == r["label"]) and (b is None or b == r["label"]):
            status[qid] = "half"          # one pass done and it agrees: train only
        else:
            status[qid] = "contradicted"
    print("t1:", Counter(status.values()))

    doc = lambda qid: qid.rsplit("-", 1)[0]
    docs = defaultdict(list)
    for qid in authors:
        docs[doc(qid)].append(qid)
    clean_docs = {d: q for d, q in docs.items() if all(status[x] == "verified" for x in q)}
    by_lang = defaultdict(list)
    for d, q in sorted(clean_docs.items()):
        by_lang[authors[q[0]]["lang"]].append(d)
    n_docs = args.dev_questions // 3
    want = {"en": round(n_docs * 0.7), "fr": n_docs - round(n_docs * 0.7)}
    dev_docs = set()
    for lang, ds in by_lang.items():
        rng.shuffle(ds)
        dev_docs.update(ds[:want.get(lang, 0)])

    pilot = [json.loads(l) for l in (T / "pilot" / "kept.jsonl").open(encoding="utf-8")]
    dev = [row(r) for r in pilot] + [row(authors[q]) for d in sorted(dev_docs) for q in docs[d]]
    teacher_train = [row(authors[q]) for q in authors if doc(q) not in dev_docs and status[q] != "contradicted"]

    v4 = defaultdict(list)
    for l in (ROOT / "data" / "train_v4.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        v4[r["source"]].append(r)
    if args.trim and not args.round2:
        raise SystemExit("--trim applies to the round-2 mixture: add --round2")
    base, keep_map = [], (V4_KEEP_R2 if args.round2 else V4_KEEP)
    if args.trim:
        keep_map = {**keep_map, **TRIM_R2}
    for src, rows in v4.items():
        keep = keep_map.get(src, None)
        rng.shuffle(rows)
        base += rows if keep is None else rows[:keep]
    round2 = []
    if args.round2:
        # More of two sources from the full mixture, never rows train_v4 already holds.
        have = {(r["state"], r.get("statement", "")) for r in base}
        extra = defaultdict(list)
        for l in (ROOT / "data" / "train.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            if r["source"] in EXTRA_R2 and (r["state"], r.get("statement", "")) not in have:
                extra[r["source"]].append(r)
        for src, n in (EXTRA_TRIM_R2 if args.trim else EXTRA_R2).items():
            rng.shuffle(extra[src])
            base += extra[src][:n]
        dev_states = {r["state"] for r in dev}
        t2 = load("author_*.jsonl", T / "t2")
        round2 = [row(r) for r in t2.values() if r["state"] not in dev_states]
        print(f"round 2: {len(round2)} teacher questions x {args.round2_repeat}, "
              f"{dict(Counter(r.get('form', '-') for r in round2 if r['kind'] == 'noul'))}")
    replay = load_replay(parse_replay(args.replay), rng)
    if replay:
        print("replay:", dict(Counter(r["source"] for r in replay)))
    train = base + teacher_train * args.teacher_repeat + round2 * args.round2_repeat + replay
    rng.shuffle(train)

    outs = [(ROOT / "data" / "dev_teacher.jsonl", dev), (ROOT / "data" / "train_teacher.jsonl", teacher_train),
            (ROOT / args.out, train)]
    for path, rows in outs:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lang = Counter(r["lang"] for r in dev)
    print(f"dev_teacher: {len(dev)} ({len(pilot)} pilot + {len(dev) - len(pilot)} t1 from {len(dev_docs)} documents), {dict(lang)}")
    print(f"train_teacher: {len(teacher_train)} questions, {dict(Counter(r['lang'] for r in teacher_train))}")
    print(f"{args.out}: {len(train)} rows = {len(base)} public + {len(teacher_train)} x {args.teacher_repeat} teacher"
          + (f" + {len(round2)} x {args.round2_repeat} round 2" if round2 else "")
          + (f" + {len(replay)} replay" if replay else ""))
    n_teacher = sum(r["source"].startswith("teacher-") for r in train)
    print(f"  teacher share: {n_teacher / len(train):.1%}")
    print("  kinds:", dict(Counter(r["kind"] for r in train)))


if __name__ == "__main__":
    main()
