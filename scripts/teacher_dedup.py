"""Filter a teacher round for duplicates and leaks, then write the solvers' blind inputs.

Checks, in author order, so the later of two near-identical items is the one dropped:
- broken rows: missing fields, label out of range, duplicated options or levels;
- a document sharing more than --doc-jaccard of its word 5-grams with a document of an earlier
  round (pilot, t1, t2), of this round, or of the dev splits: the whole document is dropped;
- a document sharing more than --jevbench-share of its word 8-grams with a JevBench document
  (the share of the smaller one): dropped, JevBench stays clean;
- a question whose content words overlap more than --question-jaccard with an earlier question
  of the same family: dropped;
- with --balance-length, the share of Choice questions whose correct option is the longest one is
  brought back to chance (mean of 1/n) by dropping, at random, questions on the side in excess: t3's
  first wave had it at 35 % against 23 %, and the first authors given a rule against it went to 0 %;
  either way a model could learn the length instead of reading the document;
- names: a two-word name that recurs in earlier documents or JevBench (the favourites the
  briefs' name pools are there to avoid, training/teacher_catalog.py) is reported, not dropped.

Survivors go to <round>/questions/q_NN.jsonl without label, rationale or family, for the two
blind solvers; scripts/teacher_agreement.py later keeps what both solve. Every decision is in
<round>/dedup.jsonl.

    python scripts/teacher_dedup.py data/teacher/t3
    python scripts/teacher_dedup.py data/teacher/t3 --authors 1-34      # first wave only
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER_KEYS = ("id", "state", "kind", "prompt", "options", "levels", "statement")
NAME = re.compile(r"\b[A-Z][a-zà-ÿ]+ [A-Z][a-zà-ÿ'-]+\b")


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def grams(text: str, n: int) -> set[str]:
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def content(r: dict) -> set[str]:
    return {w for w in words(r.get("prompt") or r.get("statement") or "") if len(w) >= 4}


def problem(r: dict) -> str | None:
    """Why a row cannot be trained on, or None."""
    for k in ("id", "family", "state", "kind", "label"):
        if k not in r:
            return f"missing {k}"
    kind, label = r["kind"], r["label"]
    if kind == "choice":
        opts = r.get("options") or []
        if not r.get("prompt") or len(opts) < 2 or len(set(opts)) != len(opts):
            return "choice needs a prompt and 2+ distinct options"
        if not 0 <= label < len(opts):
            return "label out of range"
    elif kind == "score":
        lv = r.get("levels") or []
        if not r.get("prompt") or len(lv) < 2 or len(set(lv)) != len(lv):
            return "score needs a prompt and 2+ distinct levels"
        if not 0 <= label < len(lv):
            return "label out of range"
    elif kind == "noul":
        if not r.get("statement") or label not in (0, 1):
            return "noul needs a statement and label 0/1"
    else:
        return f"unknown kind {kind}"
    return None


def load_jsonl(paths) -> list[dict]:
    return [json.loads(l) for p in paths for l in Path(p).open(encoding="utf-8") if l.strip()]


def brief_people(brief: Path) -> set[str]:
    m = re.search(r"use ONLY names from this list: (.+?)\. For organisations", brief.read_text(encoding="utf-8"))
    return set(m.group(1).split(", ")) if m else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("round_dir")
    ap.add_argument("--authors", help="range of author numbers, e.g. 1-34 (default: every author file)")
    ap.add_argument("--doc-jaccard", type=float, default=0.15)
    ap.add_argument("--jevbench-share", type=float, default=0.02)
    ap.add_argument("--question-jaccard", type=float, default=0.7)
    ap.add_argument("--balance-length", action="store_true",
                    help="drop Choice questions answered by their longest option down to the chance rate")
    args = ap.parse_args()
    rd = Path(args.round_dir)
    teacher = rd.parent

    files = sorted(rd.glob("author_*.jsonl"))
    if args.authors:
        lo, hi = map(int, args.authors.split("-"))
        files = [f for f in files if lo <= int(f.stem.split("_")[1]) <= hi]

    # Reference documents and questions: earlier rounds (everything, kept or not) and dev splits.
    earlier = [f for d in sorted(teacher.iterdir()) if d.is_dir() and d.name != rd.name
               for f in sorted(d.glob("author_*.jsonl"))]
    earlier += [p for p in (ROOT / "data" / "dev_teacher.jsonl",) if p.exists()]
    ref_rows = load_jsonl(earlier)
    ref_docs = {r["state"]: grams(r["state"], 5) for r in ref_rows}
    ref_q = [(r.get("family") or r.get("source", "").removeprefix("teacher-"), content(r)) for r in ref_rows]
    jb = {r["state"]: grams(r["state"], 8) for r in load_jsonl([ROOT / "data" / "jevbench_eval.jsonl"])}
    name_docs = Counter(n for t in set(ref_docs) | set(jb) for n in set(NAME.findall(t)))
    favourites = {n for n, c in name_docs.items() if c >= 2}

    decisions, out_by_author, rows_by_id = [], {}, {}
    seen_docs = dict(ref_docs)                       # state -> 5-grams, grows with this round
    seen_q = list(ref_q)                             # (family, content words), grows too
    doc_verdict: dict[str, str] = {}
    for f in files:
        n = f.stem.split("_")[1]
        allowed = brief_people(rd / "briefs" / f"author_{n}.md")
        for r in load_jsonl([f]):
            qid, why = r.get("id", "?"), problem(r)
            state = r.get("state", "")
            if why is None and state not in doc_verdict:
                g5, verdict = grams(state, 5), "ok"
                for other, og in seen_docs.items():
                    if other != state and og and g5 and len(g5 & og) / len(g5 | og) > args.doc_jaccard:
                        verdict = "near-duplicate of an earlier document"
                        break
                if verdict == "ok":
                    g8 = grams(state, 8)
                    for jg in jb.values():
                        if g8 and jg and len(g8 & jg) / min(len(g8), len(jg)) > args.jevbench_share:
                            verdict = "close to a JevBench document"
                            break
                doc_verdict[state] = verdict
                seen_docs[state] = g5
            if why is None and doc_verdict[state] != "ok":
                why = doc_verdict[state]
            if why is None:
                cw, fam = content(r), r["family"]
                if len(cw) >= 5 and any(of == fam and len(ow) >= 5 and len(cw & ow) / len(cw | ow) > args.question_jaccard
                                        for of, ow in seen_q):
                    why = "near-duplicate question in the same family"
                else:
                    seen_q.append((fam, cw))
            stray = sorted((set(NAME.findall(state)) & favourites) - allowed)
            decisions.append({"id": qid, "author": n, "status": "drop" if why else "keep", "reason": why,
                              "recurring_names": stray})
            if not why:
                rows_by_id[qid] = r

    if args.balance_length:
        longest = lambda r: max(range(len(r["options"])), key=lambda i: len(r["options"][i]))
        ch = [r for r in rows_by_id.values() if r["kind"] == "choice"]
        cue = [r for r in ch if longest(r) == r["label"]]
        rate = sum(1 / len(r["options"]) for r in ch) / max(len(ch), 1)
        other = [r for r in ch if longest(r) != r["label"]]
        k, side = 0, (cue if len(cue) / max(len(ch), 1) > rate else other)
        if side is cue:
            while k < len(cue) and (len(cue) - k) / (len(ch) - k) > rate:
                k += 1
        else:
            while k < len(other) and len(cue) / (len(ch) - k) < rate:
                k += 1
        reason = ("length cue (correct option the longest too often; balanced to chance)" if side is cue else
                  "length cue (correct option the longest too rarely; balanced to chance)")
        drop = {r["id"] for r in random.Random(0).sample(side, k)}
        for d in decisions:
            if d["id"] in drop:
                d["status"], d["reason"] = "drop", reason
                del rows_by_id[d["id"]]
        print(f"length cue: {len(cue)} of {len(ch)} Choice questions answered by their longest option "
              f"(chance {100 * rate:.0f} %), {k} dropped on the {'longest' if side is cue else 'other'} side")
    for d in decisions:
        if d["status"] == "keep":
            r = rows_by_id[d["id"]]
            out_by_author.setdefault(d["author"], []).append({k: r[k] for k in SOLVER_KEYS if k in r})

    qdir = rd / "questions"
    qdir.mkdir(exist_ok=True)
    for n, rows in out_by_author.items():
        with (qdir / f"q_{n}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (rd / "dedup.jsonl").open("w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    kept = sum(d["status"] == "keep" for d in decisions)
    print(f"{len(files)} author files, {len(decisions)} questions: {kept} kept, {len(decisions) - kept} dropped")
    for reason, c in Counter(d["reason"] for d in decisions if d["reason"]).most_common():
        print(f"  {c:4d}  {reason}")
    strays = Counter(n for d in decisions for n in d["recurring_names"])
    if strays:
        print(f"  recurring names from earlier rounds or JevBench: {len(strays)} "
              f"(most frequent: {strays.most_common(5)})")
    print(f"solver inputs -> {qdir}, decisions -> {rd / 'dedup.jsonl'}")


if __name__ == "__main__":
    main()
