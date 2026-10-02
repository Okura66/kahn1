"""Train-split probe: did a system see the held-out sources' training data?

The held-out set is drawn from the test splits (RTE: validation) of BANKING77, MASSIVE, SST-5,
SciTail and RTE. A model trained on a source's train split scores higher on train sentences than
on test sentences (accuracy, and above all the probability it gives the gold answer); a model
that never saw the source scores the same on both, up to split differences, which Kahn1 measures:
it never saw any of the five (training/build_dataset.py keeps them out of training).
app_reviews is left out: its held-out items already come from its only split.

    python scripts/contamination_probe.py build        # data/contam_probe.jsonl (WSL: needs datasets)
    # then: jev_models_eval.py run --set probe (Laya, Clef-flash); eval.eval_full (Kahn1, k = 3)
    python scripts/contamination_probe.py report       # reports/CONTAMINATION_PROBE.md + .json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

D = _ROOT / "data"
R = _ROOT / "reports"
PROBE = D / "contam_probe.jsonl"
KAHN1_PREDS = R / "contam_probe_kahn1_preds.json"
REPORT_MD = R / "CONTAMINATION_PROBE.md"
REPORT_JSON = R / "contamination_probe.json"
PER_SOURCE = 2000
SOURCES = ("banking77", "massive", "sst5_eval", "scitail_eval", "rte_eval")


def build() -> None:
    from datasets import load_dataset

    test = [json.loads(l) for l in (D / "eval.jsonl").open(encoding="utf-8")]
    seen = {it["state"].strip() for it in test}
    proto = {s: next(it for it in test if it["source"] == s) for s in SOURCES}
    rng = random.Random(0)
    out = []

    def add(source, rows):
        rows = [r for r in rows if r["state"].strip() not in seen]
        rng.shuffle(rows)
        for r in rows[:PER_SOURCE]:
            out.append({**r, "source": source + ":train"})

    p = proto["banking77"]
    ds = load_dataset("mteb/banking77", split="train")
    add("banking77", [{"state": r["text"], "kind": "choice", "prompt": p["prompt"], "options": p["options"],
                       "label": p["options"].index(r["label_text"])} for r in ds])
    p = proto["massive"]
    ds = load_dataset("SetFit/amazon_massive_intent_en-US", split="train")
    add("massive", [{"state": r["text"], "kind": "choice", "prompt": p["prompt"], "options": p["options"],
                     "label": p["options"].index(r["label_text"])} for r in ds])
    p = proto["sst5_eval"]
    ds = load_dataset("SetFit/sst5", split="train")
    add("sst5_eval", [{"state": r["text"], "kind": "score", "prompt": p["prompt"], "levels": p["levels"],
                       "label": r["label"]} for r in ds])
    ds = load_dataset("allenai/scitail", "snli_format", split="train")
    add("scitail_eval", [{"state": r["sentence1"], "kind": "noul", "statement": r["sentence2"],
                          "label": int(r["gold_label"] == "entailment"), "form": "statement"} for r in ds])
    ds = load_dataset("nyu-mll/glue", "rte", split="train")
    add("rte_eval", [{"state": r["sentence1"], "kind": "noul", "statement": r["sentence2"],
                      "label": int(r["label"] == 0), "form": "statement"} for r in ds])
    with PROBE.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} probe items -> {PROBE}")


def report() -> None:
    from scripts.choice_fairness import eight
    from scripts.jev_holdout import jev_outcome
    from scripts.jev_models_eval import SYSTEMS, load_preds, preds_path

    test = [json.loads(l) for l in (D / "eval.jsonl").open(encoding="utf-8")]
    probe = [json.loads(l) for l in PROBE.open(encoding="utf-8")]

    def p_gold(it: dict, a: dict) -> float:
        """Probability a Jev-format answer gives the gold answer (renormalised)."""
        if it["kind"] == "noul":
            p = float(a.get("noul", 0.0))
            return p if it["label"] == 1 else 1 - p
        probs = a.get("probabilities") or {}
        key = it["options"][it["label"]] if it["kind"] == "choice" else str(it["label"])
        s = sum(float(v) for v in probs.values()) or 1.0
        return float(probs.get(key, 0.0)) / s

    def jev_rows(items, preds, noul_sup=None):
        rows = []
        for i, it in enumerate(items):
            if i not in preds or it["source"].split(":")[0] not in SOURCES:
                continue
            a = (noul_sup or {}).get(i, preds[i])
            view = eight(it) if it["kind"] == "choice" else it
            rows.append((it["source"], jev_outcome(view, a)[0], p_gold(view, a)))
        return rows

    def kahn1_rows(items, preds):
        return [(it["source"], preds["corrects"][i], preds["all_probs"][i][preds["all_labels"][i]])
                for i, it in enumerate(items) if it["source"].split(":")[0] in SOURCES]

    systems = {"Kahn1 4B": (kahn1_rows(test, json.loads((R / "eval_v7_preds.json").read_text(encoding="utf-8"))),
                            kahn1_rows(probe, json.loads(KAHN1_PREDS.read_text(encoding="utf-8")))
                            if KAHN1_PREDS.exists() else [])}
    for s in SYSTEMS:
        systems[s] = (jev_rows(test, load_preds(preds_path(s, "heldout"))),
                      jev_rows(probe, load_preds(preds_path(s, "probe"))))

    res = {}
    for name, (t_rows, p_rows) in systems.items():
        agg = defaultdict(lambda: [0, 0, 0.0])
        for src, c, pg in t_rows + p_rows:
            a = agg[src]
            a[0] += 1; a[1] += c; a[2] += math.log(max(pg, 1e-6))
        res[name] = {src: {"n": n, "acc": c / n, "mean_logp_gold": lp / n} for src, (n, c, lp) in agg.items()}
    REPORT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")

    names = [n for n in res if any(k.endswith(":train") for k in res[n])]
    L = ["# Train-split probe", "",
         "Accuracy and mean log-probability of the gold answer on test items (the held-out set) and on",
         f"up to {PER_SOURCE} train items per source, same prompts, same 8 options. A gap well above Kahn1's (which",
         "never saw these sources) means the system saw the train split.", "",
         "| Source | " + " | ".join(f"{n} test | {n} train | gap" for n in names) + " |",
         "|---|" + "---:|---:|---:|" * len(names)]
    for metric, fmt in (("acc", lambda x: f"{100 * x:.1f} %"), ("mean_logp_gold", lambda x: f"{x:.3f}")):
        L.append(f"| **{metric}** |" + " | | |" * len(names))
        for s in SOURCES:
            cells = []
            for n in names:
                t, p = res[n].get(s), res[n].get(s + ":train")
                if t and p:
                    gap = p[metric] - t[metric]
                    cells += [fmt(t[metric]), fmt(p[metric]),
                              f"{100 * gap:+.1f}" if metric == "acc" else f"{gap:+.3f}"]
                else:
                    cells += ["", "", ""]
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "Item counts (test / train): " + ", ".join(
        f"{s} {res[names[0]][s]['n']} / {res[names[0]].get(s + ':train', {}).get('n', 0)}" for s in SOURCES) + ".", ""]
    L += base_section()
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def base_section() -> list[str]:
    """Gain over the base model on the held-out set (k = 1, no calibration; scripts/eval_base.py)."""
    files = {"Qwen3.5-4B (base)": "qwen35_4b", "Kahn1 4B, k = 1": "kahn1_4b", "Qwen3.5-9B (base, fp8)": "qwen35_9b_fp8"}
    res = {}
    for name, k in files.items():
        f = R / "base_eval" / f"heldout_full_{k}.json"
        if f.exists():
            res[name] = json.loads(f.read_text(encoding="utf-8"))["heldout_sample"]
    om = R / "open_decision_models.json"
    if len(res) < 3 or not om.exists():
        return []
    clef = json.loads(om.read_text(encoding="utf-8"))["heldout"]
    groups = ["all", "kind:choice", "kind:score", "kind:noul", "banking77", "massive"]
    L = ["## Gain over the base model", "",
         "Full held-out set, k = 1, no calibration. Base models are read with Kahn1's prompt format (qwen3 template),",
         "the 9B in fp8 (vLLM); Clef-flash is its held-out run above (int8, its own head, Noul in the supported wording).", "",
         "| Group | " + " | ".join(list(res) + ["Clef-flash"]) + " |", "|---|" + "---:|" * (len(res) + 1)]
    for g in groups:
        L.append(f"| {g} | " + " | ".join(f"{100 * res[n][g]['acc']:.1f} %" for n in res) +
                 f" | {100 * clef[g]['clef-flash']['acc']:.1f} % |")

    def cut(a, b):
        return 100 * (1 - (1 - b) / (1 - a))
    q4, k1, q9 = (res[n] for n in files)
    for src in ("banking77", "massive"):
        L.append("")
        L.append(f"{src}: Clef-flash removes {cut(q9[src]['acc'], clef[src]['clef-flash']['acc']):.0f} % of Qwen3.5-9B's errors; "
                 f"Kahn1 4B removes {cut(q4[src]['acc'], k1[src]['acc']):.0f} % of Qwen3.5-4B's (Kahn1 never saw {src}).")
    L += ["", "A gain that large on intents is what training on intent data gives, these datasets or close ones; the probe",
          "and this check cannot tell which.", ""]
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "report"))
    build() if ap.parse_args().cmd == "build" else report()
