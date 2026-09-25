"""Long-document and rule-application sources for the v4 mixture, plus a dev split.

Kahn1 v1-v3 trained on short states (median 31 tokens, max ~800): intents, reviews, NLI
sentences. JevBench-style decisions read long policies and documents. This adds, from train
splits only:

- docnli   DocNLI (BSD): document-level entailment, premises of 1.5-14k characters -> Noul
- quality  QuALITY (CC BY 4.0): multiple choice over a long article -> Choice (fixed options)
- sharc    ShARC (CC BY-SA 3.0): a rule text, a user scenario and question -> Choice among
           yes / no / not covered / more information needed (fixed options)
- wanli    WANLI (CC BY 4.0): hard NLI; entailment -> yes, neutral and contradiction -> no

ContractNLI is left out: CC BY-NC-SA, incompatible with Kahn1's open licence.

    python training/build_longdoc.py                  # data/train_longdoc.jsonl, data/dev_v4.jsonl
    python training/build_longdoc.py --mix            # + data/train_v4.jsonl (with a subsample of data/train.jsonl)

Nothing here reads data/eval.jsonl or JevBench, except to drop exact overlaps with them.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SHARC_OPTIONS = ["Yes", "No", "The rule text does not cover this question",
                 "More information is needed from the user"]


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def _choice(state, prompt, options, label, source):
    return {"state": state, "kind": "choice", "prompt": prompt, "options": options, "label": label,
            "source": source, "fixed_options": True}


def _noul(state, statement, label, source):
    return {"state": state, "kind": "noul", "statement": statement, "label": label, "source": source,
            "form": "statement"}


def _balanced(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """Noul rows balanced 50/50, at most cap."""
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(len(pos), len(neg), cap // 2)
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def docnli(split: str, cap: int, rng) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tasksource/doc-nli", split=split)
    lens = [len(p) for p in ds["premise"]]
    idx = [i for i, n in enumerate(lens) if 1500 <= n <= 14000]
    rng.shuffle(idx)
    rows = [_noul(ds[i]["premise"], ds[i]["hypothesis"], 1 if ds[i]["label"] == "entailment" else 0,
                  "docnli") for i in idx[: cap * 6]]
    return _balanced(rows, cap, rng)


def quality(split: str, cap: int, rng, max_chars: int = 20000) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("emozilla/quality", split=split)
    rows = [_choice(e["article"], e["question"].strip(), [o.strip() for o in e["options"]], int(e["answer"]),
                    "quality") for e in ds if len(e["article"]) <= max_chars]
    rng.shuffle(rows)
    return rows[:cap]


def sharc(split: str, cap: int, rng) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tasksource/sharc", split=split)
    by_label: dict[int, list[dict]] = {}
    for e in ds:
        parts = [f"Rule text:\n{(e['snippet'] or '').strip()}"]
        if (e["scenario"] or "").strip():
            parts.append(f"User's situation: {e['scenario'].strip()}")
        if (e["history"] or "").strip():
            parts.append(f"Earlier questions and answers:\n{e['history'].strip()}")
        parts.append(f"User's question: {(e['question'] or '').strip()}")
        row = _choice("\n\n".join(parts), "Based only on the rule text, what is the answer to the user's question?",
                      SHARC_OPTIONS, int(e["label"]), "sharc")
        by_label.setdefault(int(e["label"]), []).append(row)
    per = cap // 4
    out = []
    for lab, rows in sorted(by_label.items()):
        rng.shuffle(rows)
        out += rows[:per] if lab != 2 else rows[: per * 2]  # 'not covered' is rare: keep more of it
    rng.shuffle(out)
    return out[:cap]


def wanli(split: str, cap: int, rng) -> list[dict]:
    import urllib.request
    url = f"https://huggingface.co/datasets/alisawuffles/WANLI/resolve/main/{split}.jsonl"
    with urllib.request.urlopen(url, timeout=120) as r:
        lines = r.read().decode("utf-8").splitlines()
    rows = []
    for line in lines:
        e = json.loads(line)
        rows.append(_noul(e["premise"], e["hypothesis"], 1 if e["gold"] == "entailment" else 0, "wanli"))
    return _balanced(rows, cap, rng)


SOURCES = {
    # name: (loader, train split, train cap, dev split, dev cap)
    "docnli": (docnli, "train", 3000, "validation", 150),
    "quality": (quality, "train", 2000, "validation", 150),
    "sharc": (sharc, "train", 4000, "validation", 150),
    "wanli": (wanli, "train", 4000, "test", 150),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mix", action="store_true", help="also write data/train_v4.jsonl")
    ap.add_argument("--base-per-kind", default="choice=5000,score=5000,noul=2000",
                    help="how many items per kind to keep from data/train.jsonl in the mix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    banned = set()
    for path in (DATA / "eval.jsonl", DATA / "jevbench_eval.jsonl"):
        for line in path.open(encoding="utf-8"):
            e = json.loads(line)
            banned.add(_norm(e["state"]))

    train, dev = [], []
    for name, (fn, tr_split, tr_cap, dv_split, dv_cap) in SOURCES.items():
        tr = [r for r in fn(tr_split, tr_cap, rng) if _norm(r["state"]) not in banned]
        dv = [r for r in fn(dv_split, dv_cap, rng) if _norm(r["state"]) not in banned]
        tr_states = {_norm(r["state"]) for r in tr}
        dv = [r for r in dv if _norm(r["state"]) not in tr_states]
        print(f"{name:8s} train {len(tr):5d}  dev {len(dv):4d}  labels {dict(Counter(r['label'] for r in tr))}")
        train += tr
        dev += dv

    # dev: the new sources plus a stratified slice of val.jsonl (the old distribution).
    val = [json.loads(l) for l in (DATA / "val.jsonl").open(encoding="utf-8")]
    by_kind: dict[str, list[dict]] = {}
    for e in val:
        by_kind.setdefault(e["kind"], []).append(e)
    for k, rows in sorted(by_kind.items()):
        random.Random(f"dev:{k}").shuffle(rows)
        dev += rows[:200]

    write = lambda p, rows: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write(DATA / "train_longdoc.jsonl", train)
    write(DATA / "dev_v4.jsonl", dev)
    print(f"train_longdoc {len(train)}  dev_v4 {len(dev)}  kinds {dict(Counter(r['kind'] for r in dev))}")

    if args.mix:
        quotas = {k: int(v) for k, v in (kv.split("=") for kv in args.base_per_kind.split(","))}
        base = [json.loads(l) for l in (DATA / "train.jsonl").open(encoding="utf-8")]
        dev_states = {_norm(r["state"]) for r in dev}
        picked = []
        for k, n in quotas.items():
            rows = [r for r in base if r["kind"] == k and _norm(r["state"]) not in dev_states]
            random.Random(f"mix:{k}:{args.seed}").shuffle(rows)
            picked += rows[:n]
        mix = picked + train
        random.Random(args.seed).shuffle(mix)
        write(DATA / "train_v4.jsonl", mix)
        print(f"train_v4 {len(mix)}  kinds {dict(Counter(r['kind'] for r in mix))}  "
              f"sources {dict(Counter(r['source'] for r in mix).most_common(8))}")


if __name__ == "__main__":
    main()
