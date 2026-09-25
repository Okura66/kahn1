"""Teacher questions -> a hard dev split and the v5 training mixture.

- Verified = both blind solvers gave the author's label (scripts/teacher_agreement.py). A
  question a solver contradicted is dropped everywhere; unverified questions (no solver pass,
  about 1 % label noise going by the checked ones) are used for training only.
- dev: the pilot's kept questions plus whole documents drawn from the verified t1 ones,
  English and French in proportion, so no document is on both sides.
- train: the other t1 questions, repeated --teacher-repeat times (every pass permutes the
  options), mixed with a subsample of data/train_v4.jsonl that cuts short classification.

    python training/build_teacher_mix.py --dev-questions 200 --teacher-repeat 4
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
V4_KEEP = {"sharc": 2500, "wanli": 2500, "docnli": 2000, "quality": None, "clinc150": None,
           "amazon_reviews": 700, "yelp_full": 700, "go_emotions": 600, "ag_news": 500, "dbpedia_14": 500,
           "imdb": 500, "tweet_sentiment": 400}
KEYS = ("state", "kind", "prompt", "options", "levels", "statement", "label")


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
    base = []
    for src, rows in v4.items():
        keep = V4_KEEP.get(src, None) if src in V4_KEEP else None
        rng.shuffle(rows)
        base += rows if keep is None else rows[:keep]
    train = base + teacher_train * args.teacher_repeat
    rng.shuffle(train)

    for name, rows in (("dev_teacher.jsonl", dev), ("train_teacher.jsonl", teacher_train), ("train_v5.jsonl", train)):
        with (ROOT / "data" / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lang = Counter(r["lang"] for r in dev)
    print(f"dev_teacher: {len(dev)} ({len(pilot)} pilot + {len(dev) - len(pilot)} t1 from {len(dev_docs)} documents), {dict(lang)}")
    print(f"train_teacher: {len(teacher_train)} questions, {dict(Counter(r['lang'] for r in teacher_train))}")
    print(f"train_v5: {len(train)} rows = {len(base)} from train_v4 + {len(teacher_train)} x {args.teacher_repeat} teacher")
    print("  kinds:", dict(Counter(r["kind"] for r in train)))


if __name__ == "__main__":
    main()
