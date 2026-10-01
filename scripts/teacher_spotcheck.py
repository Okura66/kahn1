"""Spot check of a teacher round by one solver of another model, instead of two same-model solvers.

In pilot + t1 the two blind Opus solvers rejected 1 of 600 questions: either the labels are clean
or a same-model solver shares the author's blind spots. Round 3 instead samples whole documents
at random (English and French in proportion), has one solver of a different model answer them
blind, and measures the disagreement rate. Documents whose three questions all agree become the
round's own dev split (data/dev_teacher_t3.jsonl, via build_teacher_mix.py --round3); a question
the solver contradicts is kept out of training.

    python scripts/teacher_spotcheck.py sample data/teacher/t3 --docs 50 --chunks 2
        -> <round>/spot/q_spot_<k>.jsonl (blind inputs, the solver brief's format)
    python scripts/teacher_spotcheck.py score data/teacher/t3
        -> <round>/spotcheck.jsonl ({"id", "label", "answer", "agree"}) and the rates per family
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SOLVER_KEYS = ("id", "state", "kind", "prompt", "options", "levels", "statement")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def kept_rows(rd: Path) -> list[dict]:
    keep = {d["id"] for d in jsonl(rd / "dedup.jsonl") if d["status"] == "keep"}
    return [r for f in sorted(rd.glob("author_*.jsonl")) for r in jsonl(f) if r["id"] in keep]


def doc_of(qid: str) -> str:
    return qid.rsplit("-", 1)[0]


def sample(args) -> None:
    rd = Path(args.round_dir)
    docs = defaultdict(list)
    for r in kept_rows(rd):
        docs[doc_of(r["id"])].append(r)
    whole = {d: rs for d, rs in docs.items() if len(rs) == 3}          # all three questions survived dedup
    by_lang = defaultdict(list)
    for d, rs in sorted(whole.items()):
        by_lang[rs[0]["lang"]].append(d)
    rng = random.Random(args.seed)
    total = sum(len(v) for v in by_lang.values())
    picked = []
    for lang, ds in sorted(by_lang.items()):
        rng.shuffle(ds)
        picked += ds[:round(args.docs * len(ds) / total)]
    rng.shuffle(picked)
    out = rd / "spot"
    out.mkdir(exist_ok=True)
    for k in range(args.chunks):
        rows = [r for d in picked[k::args.chunks] for r in whole[d]]
        with (out / f"q_spot_{k + 1}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({key: r[key] for key in SOLVER_KEYS if key in r}, ensure_ascii=False) + "\n")
        print(f"{out / f'q_spot_{k + 1}.jsonl'}: {len(rows)} questions")
    print(f"{len(picked)} documents of {len(whole)} whole ones, {dict(Counter(whole[d][0]['lang'] for d in picked))}")


def score(args) -> None:
    rd = Path(args.round_dir)
    labels = {r["id"]: r for f in sorted(rd.glob("author_*.jsonl")) for r in jsonl(f)}
    answers = {a["id"]: a for f in sorted((rd / "spot").glob("answers_spot_*.jsonl")) for a in jsonl(f)}
    asked = [r["id"] for f in sorted((rd / "spot").glob("q_spot_*.jsonl")) for r in jsonl(f)]
    out, fam, kind = [], defaultdict(Counter), defaultdict(Counter)
    for qid in asked:
        r, a = labels[qid], answers.get(qid)
        if a is None:
            continue
        agree = a["answer"] == r["label"]
        out.append({"id": qid, "label": r["label"], "answer": a["answer"], "agree": agree,
                    "confidence": a.get("confidence"), "note": a.get("note", "")})
        for c, key in ((fam, r["family"]), (kind, r["kind"]), (fam, "all")):
            c[key]["agree" if agree else "disagree"] += 1
    with (rd / "spotcheck.jsonl").open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(out)} of {len(asked)} sampled questions answered")
    for name, c in [*sorted(fam.items()), *sorted(kind.items())]:
        n = sum(c.values())
        print(f"  {name:18s} {n:4d}  disagreement {100 * c['disagree'] / n:5.1f} %")
    for o in out:
        if not o["agree"]:
            print(f"  {o['id']}: author {o['label']}, solver {o['answer']} ({o['confidence']}) {o['note'][:110]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("round_dir")
    s.add_argument("--docs", type=int, default=50)
    s.add_argument("--chunks", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("round_dir")
    args = ap.parse_args()
    (sample if args.cmd == "sample" else score)(args)


if __name__ == "__main__":
    main()
