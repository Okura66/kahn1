"""Agreement filter for teacher-written decision questions (the JevK5 recipe).

An author agent writes each question with its intended label; two solver agents answer it
blind (questions_part*.jsonl carry no label or rationale). A question is kept when both
solvers give the author's label.

    python scripts/teacher_agreement.py data/teacher/pilot
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(pattern: str, d: Path) -> dict[str, dict]:
    return {r["id"]: r for f in sorted(d.glob(pattern)) for r in map(json.loads, f.open(encoding="utf-8"))}


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/teacher/pilot")
    authors, A, B = load("author_*.jsonl", d), load("answers_A_*.jsonl", d), load("answers_B_*.jsonl", d)
    by = defaultdict(Counter)
    kept, disputed = [], []
    for qid, r in authors.items():
        a, b = A.get(qid, {}).get("answer"), B.get(qid, {}).get("answer")
        if a is None or b is None:
            by["pending"]["pending"] += 1
            continue
        status = ("keep" if a == b == r["label"] else
                  "solvers_agree_against_author" if a == b else
                  "split")
        for g in (r["family"], "kind:" + r["kind"], "lang:" + r.get("lang", "?"), "all"):
            by[g][status] += 1
        (kept if status == "keep" else disputed).append((qid, r, a, b, status))
    print(f"{len(authors)} questions, {len(A)} A answers, {len(B)} B answers\n")
    print(f"{'group':28} {'n':>4} {'keep':>5} {'agree!=author':>13} {'split':>6}")
    print(f"pending (not yet answered by both): {by.pop('pending', {}).get('pending', 0)}")
    for g in sorted(by, key=lambda g: (g == "all", ":" in g, g)):
        c = by[g]
        n = sum(c.values())
        print(f"{g:28} {n:>4} {c['keep']:>5} {c['solvers_agree_against_author']:>13} {c['split']:>6}"
              f"   {100 * c['keep'] / n:.0f} %")
    print("\nDisputed:")
    for qid, r, a, b, status in disputed:
        print(f"  {qid:12} {r['kind']:6} author={r['label']} A={a} B={b}  {status}"
              f"  | A: {A.get(qid, {}).get('note', '')[:90]}")
    out = d / "kept.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for _, r, *_ in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(kept)} kept -> {out}")


if __name__ == "__main__":
    main()
