"""Open decision model pages of kahn1.com (EN + FR), built by scripts/build_kb_pages.py.

- the method and data page: every system on the same items, how each was run, the known biases,
  the train-split probe and the base-model check, per-item downloads;
- one page per measured competitor, against Kahn1 4B: Clef-flash, Laya.

Every figure is read from the reports (reports/open_decision_models.json, contamination_probe.json,
base_eval/heldout_full_*.json), so the pages and the reports cannot drift apart.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports"
D = REPO / "data"
N = "&nbsp;"

OM = json.loads((R / "open_decision_models.json").read_text(encoding="utf-8"))
PROBE = json.loads((R / "contamination_probe.json").read_text(encoding="utf-8")) \
    if (R / "contamination_probe.json").exists() else {}
BASE = {k: json.loads((R / "base_eval" / f"heldout_full_{k}.json").read_text(encoding="utf-8"))
        for k in ("qwen35_4b", "qwen35_9b_fp8", "kahn1_4b") if (R / "base_eval" / f"heldout_full_{k}.json").exists()}

# report keys -> display names
H = {"k1": "Kahn1 4B", "jev": "JEV 1.13.0", "clef": "clef-flash", "laya": "laya", "tev": "tev1"}
J = {"k1": "Kahn1 4B", "jev": "Jev 1.13.0", "k5": "JevK5 v0.2", "clef": "clef-flash", "laya": "laya", "layal": "laya-long",
     "tev": "tev1"}
NAME = {"k1": "Kahn1 4B", "jev": "JEV 1.13.0", "k5": "JevK5 v0.2", "clef": "Clef-flash", "laya": "Laya", "tev": "Tev1 4B",
        "layal": "Laya, multilingual, 8k"}
SRC = ["banking77", "massive", "sst5_eval", "app_reviews_eval", "scitail_eval", "rte_eval"]
SRC_NAME = {"banking77": "BANKING77", "massive": "MASSIVE", "sst5_eval": "SST-5", "app_reviews_eval": "App reviews",
            "scitail_eval": "SciTail", "rte_eval": "RTE"}
KIND_NAME = {"en": {"all": "All", "kind:choice": "Choice", "kind:score": "Score", "kind:noul": "Noul"},
             "fr": {"all": "Global", "kind:choice": "Choice", "kind:score": "Score", "kind:noul": "Noul"}}


def pct(x: float | None, fr: bool, bold: bool = False) -> str:
    if x is None:
        return ""
    s = f"{100 * x:.1f}".replace(".", ",") + f"{N}%" if fr else f"{100 * x:.1f}%"
    return f"<b>{s}</b>" if bold else s


def num(x: float, fr: bool, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if fr else s


def intf(n: int, fr: bool) -> str:
    return f"{n:,}".replace(",", N if fr else ",")


def pval(p: float, fr: bool) -> str:
    if p < 1e-3:
        e = f"{p:.0e}".replace("e-0", "e-").replace("e-", "e-")
        m, ex = e.split("e-")
        s = f"{m}{N}×{N}10<sup>−{ex}</sup>"
    else:
        s = f"{p:.2f}"
    return s.replace(".", ",") if fr else s


def held(key: str, group: str) -> float | None:
    return OM["heldout"][group][H[key]]["acc"]


def held_ece(key: str) -> float:
    return OM["heldout"]["all"][H[key]]["ece"]


def jb(key: str, tier: str) -> float:
    return OM["jevbench"][tier][J[key]]


def best_row(vals: list[float | None]) -> int:
    return max(range(len(vals)), key=lambda i: -1 if vals[i] is None else vals[i])


def row_pcts(vals: list[float | None], fr: bool) -> list[str]:
    b = best_row(vals)
    return [pct(v, fr, i == b) for i, v in enumerate(vals)]


# ---------------------------------------------------------------------------------------------
# shared tables
# ---------------------------------------------------------------------------------------------
def heldout_table(keys: list[str], fr: bool, groups: list[str] | None = None) -> tuple[list[str], list[list[str]]]:
    groups = groups or ["all", "kind:choice", "kind:score", "kind:noul"] + SRC
    head = ["Groupe" if fr else "Group", "Exemples" if fr else "Items"] + [NAME[k] for k in keys]
    rows = []
    for g in groups:
        label = KIND_NAME["fr" if fr else "en"].get(g) or SRC_NAME[g]
        n = OM["heldout"][g][H[keys[0]]]["n"]
        rows.append([label, intf(n, fr)] + row_pcts([held(k, g) for k in keys], fr))
    return head, rows


def jevbench_table(keys: list[str], fr: bool) -> tuple[list[str], list[list[str]]]:
    head = ["Niveau" if fr else "Tier", "Exemples" if fr else "Items"] + [NAME[k] for k in keys]
    names = {"all": ("Tous", "All"), "original": ("Standard", "Standard"), "easy": ("Facile", "Easy"),
             "hard": ("Difficile", "Hard")}
    rows = []
    for t in ("all", "original", "easy", "hard"):
        rows.append([names[t][0 if fr else 1], str(OM["jevbench"][t]["n"])] + row_pcts([jb(k, t) for k in keys], fr))
    return head, rows


def mc_held(key: str) -> tuple[int, int, float]:
    x, y, p, _ = OM["heldout_mcnemar"][H[key]]
    return x, y, p


def mc_jb(key: str) -> tuple[int, int, float]:
    x, y, p = OM["jevbench_mcnemar"][J[key]]
    return x, y, p


# ---------------------------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------------------------
def write_csvs(docs: Path) -> None:
    """Per-item outcomes: data/open_models_heldout.csv and data/open_models_jevbench.csv."""
    import sys
    sys.path[:0] = [str(REPO), str(REPO / "src")]
    from scripts.choice_fairness import eight
    from scripts.jev_holdout import jev_outcome
    from scripts.jev_models_eval import jevbench_outcome, jevbench_tasks, load_preds, preds_path

    if not (D / "eval.jsonl").exists():
        return
    items = [json.loads(l) for l in (D / "eval.jsonl").open(encoding="utf-8")]
    k1 = json.loads((R / "eval_v7_preds.json").read_text(encoding="utf-8"))

    def jl(p):
        return {r["i"]: r for r in map(json.loads, p.open(encoding="utf-8"))}

    j8, jf, js = jl(D / "jev_eval_preds_choice8.jsonl"), jl(D / "jev_eval_preds.jsonl"), \
        jl(D / "jev_eval_preds_noul_supported.jsonl")
    sysp = {s: (load_preds(preds_path(s, "heldout")), load_preds(preds_path(s, "heldout", "supported")))
            for s in ("clef-flash", "tev1", "laya")}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "source", "kind", "kahn1_4b_correct", "kahn1_4b_confidence", "jev_correct", "jev_confidence",
                "clef_flash_correct", "clef_flash_confidence", "tev1_correct", "tev1_confidence", "laya_correct",
                "laya_confidence"])
    for i, it in enumerate(items):
        view = eight(it) if it["kind"] == "choice" else it
        ja = j8[i]["answer"] if it["kind"] == "choice" else js[i]["answer"] if it["kind"] == "noul" else jf[i]["answer"]
        row = [i, it["source"], it["kind"], k1["corrects"][i], round(k1["confidences"][i], 4)]
        c, p = jev_outcome(view, ja)[:2]
        row += [c, round(p, 4)]
        for s in ("clef-flash", "tev1", "laya"):
            base, sup = sysp[s]
            a = sup[i] if it["kind"] == "noul" and i in sup else base[i]
            c, p = jev_outcome(view, a)[:2]
            row += [c, round(p, 4)]
        w.writerow(row)
    (docs / "data" / "open_models_heldout.csv").write_bytes(buf.getvalue().encode("utf-8"))

    jbi = [json.loads(l) for l in (D / "jevbench_eval.jsonl").open(encoding="utf-8")]
    tasks = jevbench_tasks()
    k1j = json.loads((R / "jevbench_v7_preds.json").read_text(encoding="utf-8"))["corrects"]
    clef, laya = load_preds(preds_path("clef-flash", "jevbench")), load_preds(preds_path("laya", "jevbench"))
    tev = load_preds(preds_path("tev1", "jevbench"))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["task_id", "tier", "kind", "kahn1_4b_correct", "clef_flash_correct", "tev1_correct", "laya_correct"])
    for i, it in enumerate(jbi):
        t = tasks[it["id"]]
        w.writerow([it["id"], it["source"].removeprefix("jevbench-"), it["kind"], k1j[i],
                    jevbench_outcome(t, clef[i])[0], jevbench_outcome(t, tev[i])[0], jevbench_outcome(t, laya[i])[0]])
    (docs / "data" / "open_models_jevbench.csv").write_bytes(buf.getvalue().encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# probe and base-model check
# ---------------------------------------------------------------------------------------------
PROBE_SYS = [("k1", "Kahn1 4B"), ("clef", "clef-flash"), ("tev", "tev1"), ("laya", "laya")]
PROBE_SRC = ["banking77", "massive", "sst5_eval", "scitail_eval", "rte_eval"]


def probe_table(fr: bool) -> str:
    from scripts.build_kb_pages import table
    if not PROBE:
        return ""
    keys = [(k, n) for k, n in PROBE_SYS if n in PROBE and any(s.endswith(":train") for s in PROBE[n])]
    head = ["Source"] + [f"{NAME[k]}, {'test → train' if not fr else 'test → train'}" for k, _ in keys]
    rows = []
    for s in PROBE_SRC:
        cells = [SRC_NAME[s]]
        for k, n in keys:
            t, p = PROBE[n][s], PROBE[n][s + ":train"]
            gap = 100 * (p["acc"] - t["acc"])
            g = (f"{gap:+.1f}".replace(".", ",") if fr else f"{gap:+.1f}") + " pt"
            cells.append(f"{pct(t['acc'], fr)} → {pct(p['acc'], fr)} ({g})")
        rows.append(cells)
    return table(head, rows, set(range(1, len(keys) + 1)))


def base_rows() -> dict:
    """Held-out accuracy at k = 1, no calibration: bases vs fine-tuned (Kahn1 from eval_base, Clef-flash ours)."""
    out = {}
    for k, res in BASE.items():
        h = res["heldout_sample"]
        out[k] = {g: h[g]["acc"] for g in ["all", "kind:choice", "kind:score", "kind:noul"] + SRC if g in h}
    return out


def base_table(fr: bool) -> str:
    from scripts.build_kb_pages import table
    b = base_rows()
    if not {"qwen35_4b", "qwen35_9b_fp8"} <= set(b):
        return ""
    cols = [("qwen35_4b", "Qwen3.5-4B" + (" (base)" if not fr else " (base)")),
            ("kahn1_4b", "Kahn1 4B, k" + N + "=" + N + "1"),
            ("tev", "Tev1 4B"),
            ("qwen35_9b_fp8", "Qwen3.5-9B (base, fp8)"),
            ("clef", "Clef-flash")]
    head = ["Groupe" if fr else "Group"] + [c for _, c in cols]
    rows = []
    for g in ["all", "kind:choice", "kind:score", "kind:noul", "banking77", "massive"]:
        label = KIND_NAME["fr" if fr else "en"].get(g) or SRC_NAME[g]
        vals = []
        for k, _ in cols:
            vals.append(held(k, g) if k in ("clef", "tev") else b.get(k, {}).get(g))
        rows.append([label] + [pct(v, fr) for v in vals])
    return table(head, rows, {1, 2, 3, 4, 5})


# ---------------------------------------------------------------------------------------------
# the method and data page
# ---------------------------------------------------------------------------------------------
def probe_reading(fr: bool) -> str:
    """What the base-model check says, from the numbers."""
    b = base_rows()
    if not {"qwen35_4b", "qwen35_9b_fp8", "kahn1_4b"} <= set(b):
        return ""

    def err(x):
        return 1 - x

    def cut(base, tuned):
        return 1 - err(tuned) / err(base)

    cb, cm = cut(b["qwen35_9b_fp8"]["banking77"], held("clef", "banking77")), cut(b["qwen35_9b_fp8"]["massive"], held("clef", "massive"))
    kb_, km = cut(b["qwen35_4b"]["banking77"], b["kahn1_4b"]["banking77"]), cut(b["qwen35_4b"]["massive"], b["kahn1_4b"]["massive"])
    tb, tm = cut(b["qwen35_4b"]["banking77"], held("tev", "banking77")), cut(b["qwen35_4b"]["massive"], held("tev", "massive"))
    e = lambda x: pct(err(x), fr)
    q = lambda x: f"{100 * x:.0f}".replace(".", ",") + (f"{N}%" if fr else "%")
    if fr:
        return (f"Sur BANKING77, Clef-flash supprime {q(cb)} des erreurs de son modèle de base (de {e(b['qwen35_9b_fp8']['banking77'])} "
                f"à {e(held('clef', 'banking77'))} d'erreurs), et {q(cm)} sur MASSIVE ; Kahn1 4B en supprime {q(kb_)} et {q(km)} "
                f"(Kahn1 n'a vu aucun des deux), et Tev1, qui a vu BANKING77 mais pas MASSIVE, {q(tb)} et {q(tm)}. Un gain aussi fort sur les intentions est ce que donne un entraînement sur des "
                "données d'intention, qu'il s'agisse de ces datasets ou de datasets proches, et ces deux tests ne permettent pas "
                "de trancher. Lecture honnête : l'avance de Clef-flash en Choice est réelle sur ces exemples, mais on ne peut pas "
                "affirmer qu'elle est zero-shot. Les modèles de base sont lus avec le format de prompt de Kahn1, en fp8 pour le 9B.")
    return (f"On BANKING77, Clef-flash removes {q(cb)} of its base model's errors (from {e(b['qwen35_9b_fp8']['banking77'])} to "
            f"{e(held('clef', 'banking77'))} errors), and {q(cm)} on MASSIVE; Kahn1 4B removes {q(kb_)} and {q(km)} (Kahn1 saw "
            f"neither), and Tev1, which saw BANKING77 but not MASSIVE, {q(tb)} and {q(tm)}. A gain that large on intents is what training on intent data gives, whether these datasets or close "
            "ones, and these two tests cannot tell which. The honest reading: Clef-flash's lead on Choice is real on these "
            "items, but we cannot claim it is zero-shot. The base models are read with Kahn1's prompt format, the 9B in fp8.")


def open_models(fr: bool) -> tuple[str, list]:
    from scripts.build_kb_pages import hero, p, sec, table, ul
    K = ["k1", "clef", "jev", "tev", "laya"]
    KJ = ["k1", "jev", "k5", "clef", "tev", "laya"]
    tx, ty, tp = mc_held("tev")
    vx, vy, vp = mc_jb("tev")
    cx, cy, cp = mc_held("clef")
    lx, ly, lp = mc_held("laya")
    jx, jy, jp = mc_held("jev")
    bx, by, bp = mc_jb("clef")
    ax, ay, ap = mc_jb("laya")
    tr_jb = OM["meta"].get("laya_jevbench_truncated", 57)
    nb = OM["noul_bare"]
    if fr:
        items = [("apercu", "En bref"), ("holdout", "Holdout Kahn1, par dataset"), ("jevbench", "JevBench, par niveau"),
                 ("methode", "Comment chaque système a tourné"), ("biais", "Biais connus et limites"),
                 ("exposition", "Un système a-t-il vu les sources du test ?"), ("donnees", "Données et reproduction")]
        h = hero("Modèles de décision ouverts, mesurés à périmètre égal",
                 f"Kahn1 4B, Clef-flash, Tev1, Laya et JEV sur les mêmes 14{N}663 exemples réservés et les 231 exemples publics "
                 "de JevBench, sur un seul GPU : les chiffres, la méthode de test, et les biais que l'on connaît, "
                 "y compris les nôtres.", items, True, "Mis à jour le 2 octobre 2026")
        hd, rw = heldout_table(K, True, ["all", "kind:choice", "kind:score", "kind:noul"])
        s1 = sec(1, "apercu", "En bref", p(
            f"Clef-flash, un modèle de 9B, est le plus précis sur notre holdout ({pct(held('clef', 'all'), True)}), devant "
            f"JEV ({pct(held('jev', 'all'), True)}) et Kahn1 4B ({pct(held('k1', 'all'), True)}), grâce à Choice. Sur JevBench, Kahn1 4B, Jev, JevK5 et Clef-flash tiennent en 4 points, "
            "et aucun écart entre Kahn1 4B et l'un d'eux n'est significatif. Tev1 (Together AI, 4B, la même base que Kahn1) "
            f"est au niveau de Kahn1 4B sur le holdout ({pct(held('tev', 'all'), True)}, écart non significatif), le meilleur "
            f"en Noul ({pct(held('tev', 'kind:noul'), True)}) et le mieux calibré (ECE {num(held_ece('tev'), True)}), mais loin "
            f"derrière sur JevBench ({pct(jb('tev', 'all'), True)}, niveau difficile {pct(jb('tev', 'hard'), True)}). Laya, un "
            "encodeur de 421M, est loin derrière sur les deux, alors que ses données d'entraînement couvrent les six sources "
            "du holdout.") +
            table(hd, rw, {1, 2, 3, 4, 5, 6}) +
            p(f"ECE (0 est parfait) : Kahn1 4B {num(held_ece('k1'), True)}, Clef-flash {num(held_ece('clef'), True)}, "
              f"JEV {num(held_ece('jev'), True)}, Tev1 {num(held_ece('tev'), True)}, Laya {num(held_ece('laya'), True)}. Chaque système est mesuré tel qu'il est "
              "livré : Kahn1 avec sa calibration par température, Laya avec la sienne, Clef-flash en softmax brute, Tev1 par "
              "les probabilités de ses lettres d'option (qu'il ne présente pas comme une confiance), JEV tel que l'API répond."))
        hd, rw = heldout_table(K, True)
        s2 = sec(2, "holdout", "Holdout Kahn1, par dataset", p(
            f"14{N}663 exemples issus des splits de test de six datasets publics : intentions (BANKING77, MASSIVE), échelles "
            "à 5 niveaux (SST-5, avis d'applications), implication (RTE, SciTail). Une question par requête, mêmes options "
            "pour tous : chaque question Choice liste la bonne intention et 7 distracteurs tirés de l'état, 8 options en tout.") +
            table(hd, rw, {1, 2, 3, 4, 5, 6}) +
            p(f"Apparié à Kahn1 4B, exemple par exemple (test exact de McNemar) : Clef-flash réussit {intf(cy, True)} exemples "
              f"que Kahn1 rate, Kahn1 {intf(cx, True)} que Clef-flash rate (p{N}={N}{pval(cp, True)}) ; JEV {intf(jy, True)} "
              f"contre {intf(jx, True)} (p{N}={N}{pval(jp, True)}) ; Laya {intf(ly, True)} contre {intf(lx, True)} "
              f"(p{N}={N}{pval(lp, True)}) ; Tev1 {intf(ty, True)} contre {intf(tx, True)} (p{N}={N}{pval(tp, True)}). "
              "Clef-flash et JEV sont donc significativement devant Kahn1 4B sur ce holdout.",
              "La formulation de Noul change beaucoup les résultats. Posée comme la phrase brute (la forme native de Jev), "
              f"la question devient « est-ce vrai en général ? » : JEV tombe à {pct(nb['JEV 1.13.0']['acc'], True)}, "
              f"Clef-flash à {pct(nb['clef-flash']['acc'], True)}, Laya à {pct(nb['laya']['acc'], True)}. "
              "Les tableaux utilisent <code>The text supports this statement: {statement}</code>, la formulation la plus "
              "favorable aux trois, celle que Kahn1 pose lui-même."))
        hd, rw = jevbench_table(KJ, True)
        s3 = sec(3, "jevbench", "JevBench, par niveau", p(
            "<a href=\"https://github.com/fstandhartinger/jevbench\" rel=\"noopener\">JevBench</a> est un benchmark public "
            "indépendant pour les modèles de classe Jev, avec des rubriques longues et des décisions difficiles. Clef-flash, "
            "Tev1 et Laya ont reçu les questions et les états JevBench natifs, descriptions des critères comprises, comme le "
            "runner de JevBench les envoie à Jev.") + table(hd, rw, {1, 2, 3, 4, 5, 6, 7}) +
            p(f"Apparié à Kahn1 4B : Clef-flash {by} contre {bx} (p{N}={N}{pval(bp, True)}, non significatif), Laya {ay} contre "
              f"{ax} (p{N}={N}{pval(ap, True)}), Tev1 {vy} contre {vx} (p{N}={N}{pval(vp, True)}), Jev 11 contre 13 (p{N}={N}0,84), JevK5 10 contre 13 (p{N}={N}0,68). "
              "Trois exécutions sur les mêmes exemples : la nôtre pour Kahn1, Clef-flash, Tev1 et Laya, celle de JevBench pour Jev, "
              "celle des auteurs de JevK5 pour JevK5 (l'exécution de JevBench donne 85,3 % à JevK5 v0.2)."))
        s4 = sec(4, "methode", "Comment chaque système a tourné", ul([
            "<b>Mêmes exemples, mêmes options, mêmes labels.</b> Choice sur les 8 mêmes options que Kahn1, Score sur les "
            "mêmes niveaux, Noul dans la formulation ci-dessus. Les questions sont envoyées dans les champs de Jev "
            "(<code>type</code>, <code>instructions</code>, <code>criteria</code>), le format natif de Clef-flash et Laya.",
            f"<b>Kahn1 4B</b> : <code>Okura66/Kahn1-Qwen3.5-4B</code>, vLLM, bf16, k{N}={N}3 ordres d'options moyennés, "
            "calibration par température, ses prédictions enregistrées.",
            "<b>Clef-flash</b> : <code>Cloudflare/clef-flash</code> avec son code de publication "
            "(<code>joint_schema_model.py</code>), transformers, softmax par question. Poids en <b>int8</b> "
            f"(bitsandbytes) : en bf16, ses 18,8{N}Go ne tiennent pas dans les 16{N}Go de notre carte. Texte seul.",
            "<b>Tev1</b> : <code>togethercomputer/Tev1-4B-experimental</code>, vLLM, bf16, son prompt système recommandé "
            "et sa décision JSON (état, question, options étiquetées par lettres), thinking désactivé. Sa réponse est la "
            "lettre la plus probable au premier token, celle qu'il génère en greedy ; les probabilités des lettres servent "
            "de confiance. Score passe par ses options, dans l'ordre des niveaux ; Noul par deux options, oui et non.",
            "<b>Laya</b> : <code>convaiinnovations/laya</code> 0.3.23, son <code>Router</code> recommandé (il choisit le "
            "checkpoint anglais ou multilingue), sa calibration livrée.",
            "<b>JEV 1.13.0</b> : l'API TypeSafe, appelée par nous avec les mêmes exemples (holdout) ; sur JevBench, les "
            "résultats publiés par JevBench.",
            f"<b>Matériel</b> : une RTX 5070 Ti (16{N}Go), WSL2. <b>Statistiques</b> : test exact de McNemar apparié, ECE "
            "sur 15 tranches de p<sub>max</sub>. Une seule exécution par système : tous sont déterministes à "
            "l'inférence."]))
        s5 = sec(5, "biais", "Biais connus et limites", ul([
            "<b>Le holdout est le nôtre.</b> Nous avons choisi ses six sources et sa forme (8 options en Choice). Elles "
            "ont été écartées de l'entraînement de Kahn1, mais Kahn1 a été entraîné sur des tâches voisines : intentions "
            "(CLINC150, qui contient des intentions bancaires), sentiment (Yelp, Amazon, IMDB), NLI (MNLI, ANLI, WANLI…).",
            "<b>Laya a été entraîné sur les six sources du holdout</b> (BANKING77, MASSIVE, SST-5, SciTail, RTE, "
            "app_reviews, selon sa fiche) : ses chiffres de holdout sont in-distribution, pas zero-shot, et pourtant derrière.",
            "<b>Les données d'entraînement de Clef-flash ne sont pas publiées</b> (« internal synthetic datasets »). "
            "BANKING77 figure dans les benchmarks de son éditeur. Voir la section suivante.",
            "<b>Tev1 a été entraîné sur BANKING77 et SST-5</b> (et MultiNLI, BoolQ, AG News, plus des règles synthétiques, "
            "selon son <code>DATA_SOURCES.md</code>) : ses chiffres sur ces deux sources sont in-distribution. Pas sur JevBench, "
            "qu'il déclare n'avoir pas utilisé. Sa licence de poids est encore « en cours de finalisation ».",
            "<b>Clef-flash a tourné en int8</b>, pas en bf16 comme publié : ses chiffres peuvent bouger un peu en bf16.",
            f"<b>Laya lit 512 tokens</b> avec son checkpoint anglais : {tr_jb} des 231 états JevBench sont tronqués. Lu en "
            f"entier par son checkpoint multilingue (8{N}192 tokens), il fait moins bien ({pct(jb('layal', 'all'), True)}) : "
            "la troncature n'explique pas son score.",
            "<b>Réglages livrés</b> : Kahn1 moyenne 3 ordres d'options et se calibre, les autres répondent en une passe, "
            "Tev1 avec les options dans l'ordre donné. "
            "L'ECE compare donc des systèmes tels que livrés, pas après une recalibration commune.",
            "<b>Trois runners sur JevBench</b> ; seule la moitié publique de JevBench a pu être testée ici."]))
        s6 = sec(6, "exposition", "Un système a-t-il vu les sources du test ?", p(
            f"Deux tests. D'abord une sonde : 2{N}000 exemples tirés des splits <i>train</i> de cinq sources, posés "
            "exactement comme le holdout. Un modèle qui a mémorisé ces exemples fait mieux sur train que sur test ; un "
            "modèle qui n'a jamais vu la source fait pareil sur les deux. Kahn1, qui n'a vu aucune de ces sources, sert de "
            "témoin ; Noul y est posé comme la phrase brute pour Clef-flash et Laya, des deux côtés.") + probe_table(True) +
            p("Aucun système ne montre d'écart au-delà de celui du témoin, ni Laya ni Tev1, qui déclarent pourtant avoir été "
              "entraînés sur ces sources (toutes pour Laya, BANKING77 et SST-5 pour Tev1). "
              "La sonde détecte la mémorisation, pas l'exposition : un modèle entraîné sur un split train sans "
              "sur-apprentissage fait aussi bien sur le test. Elle ne prouve donc pas qu'un système n'a pas vu ces données.",
              f"Ensuite, le gain par rapport au modèle de base, à k{N}={N}1 sans calibration : ce que le "
              "fine-tuning ajoute au modèle brut.") + base_table(True) + p(probe_reading(True)))
        s7 = sec(7, "donnees", "Données et reproduction", p(
            "Résultats par exemple : <a href=\"/data/open_models_heldout.csv\">open_models_heldout.csv</a> (holdout, "
            "juste ou faux et confiance pour chaque système) et <a href=\"/data/open_models_jevbench.csv\">open_models_jevbench.csv</a>. "
            "L'index d'exemple renvoie à <code>data/eval.jsonl</code>, que "
            "<code>training/build_dataset.py</code> reconstruit à l'identique.",
            "Scripts : <code>scripts/jev_models_eval.py</code> (Clef-flash, Laya, rapport), "
            "<code>scripts/contamination_probe.py</code> (sonde), <code>scripts/eval_base.py</code> (modèles de base). "
            "Rapports : <a href=\"https://github.com/Okura66/kahn1/blob/main/reports/OPEN_DECISION_MODELS.md\" rel=\"noopener\">OPEN_DECISION_MODELS.md</a>, "
            "<a href=\"https://github.com/Okura66/kahn1/blob/main/reports/CONTAMINATION_PROBE.md\" rel=\"noopener\">CONTAMINATION_PROBE.md</a>.",
            "Comparaisons détaillées : <a href=\"/fr/comparer/kahn1-vs-clef-flash/\">Kahn1 vs Clef-flash</a>, "
            "<a href=\"/fr/comparer/kahn1-vs-laya/\">Kahn1 vs Laya</a>. Une erreur dans la façon dont un système a été "
            "lancé ? <a href=\"https://github.com/Okura66/kahn1/issues\" rel=\"noopener\">Ouvrez une issue</a>, on relance."))
        return h + s1 + s2 + s3 + s4 + s5 + s6 + s7, []
    items = [("overview", "At a glance"), ("holdout", "Kahn1 held-out set, per dataset"), ("jevbench", "JevBench, per tier"),
             ("method", "How each system was run"), ("biases", "Known biases and limits"),
             ("exposure", "Did a system see the test sources?"), ("data", "Data and reproduction")]
    h = hero("Open decision models, measured like for like",
             "Kahn1 4B, Clef-flash, Tev1, Laya and JEV on the same 14,663 held-out items and the 231 public JevBench items, on "
             "one GPU: the numbers, the test method, and the biases we know of, ours included.", items, False,
             "Updated October 2, 2026")
    hd, rw = heldout_table(K, False, ["all", "kind:choice", "kind:score", "kind:noul"])
    s1 = sec(1, "overview", "At a glance", p(
        f"Clef-flash, a 9B model, is the most accurate system on our held-out set ({pct(held('clef', 'all'), False)}), ahead "
        f"of JEV ({pct(held('jev', 'all'), False)}) and Kahn1 4B ({pct(held('k1', 'all'), False)}), on the strength of Choice. "
        "On JevBench, Kahn1 4B, Jev, JevK5 and Clef-flash are within "
        "4 points, and no gap between Kahn1 4B and any of them is significant. Tev1 (Together AI, 4B, the same base as "
        f"Kahn1) is level with Kahn1 4B on the held-out set ({pct(held('tev', 'all'), False)}, not a significant gap), the "
        f"best on Noul ({pct(held('tev', 'kind:noul'), False)}) and the best calibrated (ECE {num(held_ece('tev'), False)}), "
        f"but far behind on JevBench ({pct(jb('tev', 'all'), False)}, hard tier {pct(jb('tev', 'hard'), False)}). "
        "Laya, a 421M encoder, is far behind on both, although its training data covers the six sources of the held-out "
        "set.") + table(hd, rw, {1, 2, 3, 4, 5, 6}) +
        p(f"ECE (0 is perfect): Kahn1 4B {num(held_ece('k1'), False)}, Clef-flash {num(held_ece('clef'), False)}, "
          f"JEV {num(held_ece('jev'), False)}, Tev1 {num(held_ece('tev'), False)}, Laya {num(held_ece('laya'), False)}. Each system is measured as shipped: "
          "Kahn1 with its temperature calibration, Laya with its own, Clef-flash as a raw softmax, Tev1 through the "
          "probabilities of its option letters (which it does not present as a confidence), JEV as its API answers."))
    hd, rw = heldout_table(K, False)
    s2 = sec(2, "holdout", "Kahn1 held-out set, per dataset", p(
        "14,663 items from the test splits of six public datasets: intents (BANKING77, MASSIVE), 5-level scales (SST-5, "
        "app reviews), entailment (RTE, SciTail). One question per request, the same options for everyone: each Choice "
        "question lists the right intent and 7 distractors seeded from the state, 8 options in all.") +
        table(hd, rw, {1, 2, 3, 4, 5, 6}) +
        p(f"Paired with Kahn1 4B item by item (exact McNemar test): Clef-flash gets {intf(cy, False)} items right that Kahn1 "
          f"misses, Kahn1 {intf(cx, False)} that Clef-flash misses (p&nbsp;=&nbsp;{pval(cp, False)}); JEV {intf(jy, False)} "
          f"against {intf(jx, False)} (p&nbsp;=&nbsp;{pval(jp, False)}); Laya {intf(ly, False)} against {intf(lx, False)} "
          f"(p&nbsp;=&nbsp;{pval(lp, False)}); Tev1 {intf(ty, False)} against {intf(tx, False)} (p&nbsp;=&nbsp;{pval(tp, False)}). "
          "Clef-flash and JEV are significantly ahead of Kahn1 4B on this held-out set.",
          "How Noul is worded matters a lot. Asked as the bare statement (Jev's native form), the question turns into "
          f"\"is this true in general?\": JEV drops to {pct(nb['JEV 1.13.0']['acc'], False)}, Clef-flash to "
          f"{pct(nb['clef-flash']['acc'], False)}, Laya to {pct(nb['laya']['acc'], False)}. The tables "
          "use <code>The text supports this statement: {statement}</code>, the wording that suits all three best and the "
          "one Kahn1 asks itself."))
    hd, rw = jevbench_table(KJ, False)
    s3 = sec(3, "jevbench", "JevBench, per tier", p(
        "<a href=\"https://github.com/fstandhartinger/jevbench\" rel=\"noopener\">JevBench</a> is an independent public "
        "benchmark for Jev-class models, with long rubrics and hard judgment calls. Clef-flash, Tev1 and Laya got the native "
        "JevBench questions and states, criteria descriptions included, as JevBench's runner sends them to Jev.") +
        table(hd, rw, {1, 2, 3, 4, 5, 6, 7}) +
        p(f"Paired with Kahn1 4B: Clef-flash {by} against {bx} (p&nbsp;=&nbsp;{pval(bp, False)}, not significant), Laya {ay} "
          f"against {ax} (p&nbsp;=&nbsp;{pval(ap, False)}), Tev1 {vy} against {vx} (p&nbsp;=&nbsp;{pval(vp, False)}), Jev 11 against 13 (p&nbsp;=&nbsp;0.84), JevK5 10 against 13 "
          "(p&nbsp;=&nbsp;0.68). Three runs on the same items: ours for Kahn1, Clef-flash, Tev1 and Laya, JevBench's for Jev, "
          "JevK5's authors' for JevK5 (JevBench's own run gives JevK5 v0.2 85.3%)."))
    s4 = sec(4, "method", "How each system was run", ul([
        "<b>Same items, same options, same labels.</b> Choice over the same 8 options as Kahn1, Score over the same levels, "
        "Noul in the wording above. Questions are sent in Jev's fields (<code>type</code>, <code>instructions</code>, "
        "<code>criteria</code>), Clef-flash's and Laya's native format.",
        "<b>Kahn1 4B</b>: <code>Okura66/Kahn1-Qwen3.5-4B</code>, vLLM, bf16, k&nbsp;=&nbsp;3 option orders averaged, "
        "temperature calibration, its saved predictions.",
        "<b>Clef-flash</b>: <code>Cloudflare/clef-flash</code> through its release code (<code>joint_schema_model.py</code>), "
        "transformers, a softmax per question. Weights in <b>int8</b> (bitsandbytes): in bf16 its 18.8&nbsp;GB do not fit "
        "in our card's 16&nbsp;GB. Text only.",
        "<b>Tev1</b>: <code>togethercomputer/Tev1-4B-experimental</code>, vLLM, bf16, its recommended system prompt and "
        "JSON decision (state, question, letter-labelled options), thinking off. Its answer is the most likely letter at "
        "the first token, the one it generates greedily; the letters' probabilities serve as its confidence. Score goes "
        "through its options, in level order; Noul through two options, yes and no.",
        "<b>Laya</b>: <code>convaiinnovations/laya</code> 0.3.23, its recommended <code>Router</code> (which picks the "
        "English or the multilingual checkpoint), its shipped calibration.",
        "<b>JEV 1.13.0</b>: TypeSafe's API, called by us with the same items (held-out set); on JevBench, the outcomes "
        "JevBench publishes.",
        "<b>Hardware</b>: one RTX 5070 Ti (16&nbsp;GB), WSL2. <b>Statistics</b>: paired exact McNemar test, ECE over 15 "
        "bins of p<sub>max</sub>. One run per system: all are deterministic at inference."]))
    s5 = sec(5, "biases", "Known biases and limits", ul([
        "<b>The held-out set is ours.</b> We chose its six sources and its shape (8 options on Choice). They were kept out "
        "of Kahn1's training, but Kahn1 was trained on neighbouring tasks: intents (CLINC150, which has banking intents), "
        "sentiment (Yelp, Amazon, IMDB), NLI (MNLI, ANLI, WANLI…).",
        "<b>Laya was trained on the six sources of the held-out set</b> (BANKING77, MASSIVE, SST-5, SciTail, RTE, "
        "app_reviews, per its model card): its held-out figures are in-distribution, not zero-shot, and still behind.",
        "<b>Clef-flash's training data is not published</b> (\"internal synthetic datasets\"). BANKING77 is among its "
        "provider's benchmarks. See the next section.",
        "<b>Tev1 was trained on BANKING77 and SST-5</b> (and MultiNLI, BoolQ, AG News, plus synthetic rules, per its "
        "<code>DATA_SOURCES.md</code>): its figures on those two sources are in-distribution. Not on JevBench, which it says "
        "it did not use. Its weights licence is still \"being finalized\".",
        "<b>Clef-flash ran in int8</b>, not in bf16 as released: its figures may move a little in bf16.",
        f"<b>Laya reads 512 tokens</b> with its English checkpoint: {tr_jb} of the 231 JevBench states are truncated. Read in "
        f"full by its multilingual checkpoint (8,192 tokens), it scores lower ({pct(jb('layal', 'all'), False)}): truncation "
        "does not explain its score.",
        "<b>Shipped settings</b>: Kahn1 averages 3 option orders and calibrates, the others answer in one pass, Tev1 with "
        "the options in the order given. ECE "
        "therefore compares systems as shipped, not after a common recalibration.",
        "<b>Three runners on JevBench</b>; only JevBench's public half could be tested here."]))
    s6 = sec(6, "exposure", "Did a system see the test sources?", p(
        "Two tests. First a probe: 2,000 items drawn from the <i>train</i> splits of five sources, asked exactly like the "
        "held-out set. A model that memorised those items scores higher on train than on test; a model that never saw "
        "the source scores the same on both. Kahn1, which saw none of these sources, is the control; Noul is asked as the "
        "bare statement for Clef-flash and Laya, on both splits.") + probe_table(False) +
        p("No system shows a gap beyond the control's, neither Laya nor Tev1, which say they were trained on these sources "
          "(all of them for Laya, BANKING77 and SST-5 for Tev1). The probe detects "
          "memorisation, not exposure: a model trained on a train split without overfitting does as well on the test "
          "split. It does not prove that a system never saw this data.",
          "Then, the gain over the base model, at k&nbsp;=&nbsp;1 without calibration: what fine-tuning adds to the raw "
          "model.") + base_table(False) + p(probe_reading(False)))
    s7 = sec(7, "data", "Data and reproduction", p(
        "Per-item outcomes: <a href=\"/data/open_models_heldout.csv\">open_models_heldout.csv</a> (held-out set, right or "
        "wrong and confidence for each system) and <a href=\"/data/open_models_jevbench.csv\">open_models_jevbench.csv</a>. "
        "The item index points into <code>data/eval.jsonl</code>, which <code>training/build_dataset.py</code> rebuilds "
        "identically.",
        "Scripts: <code>scripts/jev_models_eval.py</code> (Clef-flash, Laya, report), <code>scripts/contamination_probe.py</code> "
        "(probe), <code>scripts/eval_base.py</code> (base models). Reports: "
        "<a href=\"https://github.com/Okura66/kahn1/blob/main/reports/OPEN_DECISION_MODELS.md\" rel=\"noopener\">OPEN_DECISION_MODELS.md</a>, "
        "<a href=\"https://github.com/Okura66/kahn1/blob/main/reports/CONTAMINATION_PROBE.md\" rel=\"noopener\">CONTAMINATION_PROBE.md</a>.",
        "Head to head: <a href=\"/compare/kahn1-vs-clef-flash/\">Kahn1 vs Clef-flash</a>, "
        "<a href=\"/compare/kahn1-vs-laya/\">Kahn1 vs Laya</a>. Think a system was run wrong? "
        "<a href=\"https://github.com/Okura66/kahn1/issues\" rel=\"noopener\">Open an issue</a> and we will re-run it."))
    return h + s1 + s2 + s3 + s4 + s5 + s6 + s7, []


# ---------------------------------------------------------------------------------------------
# head-to-head pages
# ---------------------------------------------------------------------------------------------
# Facts, from each model's card and release code (checked 2026-10-02). (EN, FR) per cell.
FACTS = {
    "k1": {
        "dev": ("Kahn1, independent", "Kahn1, indépendant"),
        "lic": ("Weights Apache 2.0, code MIT", "Poids Apache 2.0, code MIT"),
        "base": ("Qwen3.5-4B + LoRA, merged", "Qwen3.5-4B + LoRA, fusionné"),
        "size": ("4B; 8.4 GB in bf16", "4B ; 8,4 Go en bf16"),
        "inputs": ("Text", "Texte"),
        "read": ("Option-token logits, 3 option orders averaged, temperature calibration",
                 "Logits des tokens d'option, 3 ordres d'options moyennés, calibration par température"),
        "api": ("Jev question fields under a \"schema\" key at /v1/evaluate/jev (adapter needed for a Jev client)",
                "Champs de question de Jev sous une clé « schema » sur /v1/evaluate/jev (adaptateur pour un client Jev)"),
        "runs": ("One 16 GB GPU in bf16 (vLLM), or a CPU (transformers)",
                 "Un GPU de 16 Go en bf16 (vLLM), ou un CPU (transformers)"),
        "lat": ("Median 88.7 ms per request, k = 3, RTX 5070 Ti (our measure)",
                "Médiane 88,7 ms par requête, k = 3, RTX 5070 Ti (notre mesure)"),
    },
    "clef": {
        "dev": ("Cloudflare", "Cloudflare"),
        "lic": ("Apache 2.0", "Apache 2.0"),
        "base": ("Qwen3.5-9B + a joint schema head", "Qwen3.5-9B + une tête de schéma jointe"),
        "size": ("9B; 18.8 GB in bf16", "9B ; 18,8 Go en bf16"),
        "inputs": ("Text, JSON, images, video; up to 16,384 tokens by default",
                   "Texte, JSON, images, vidéo ; jusqu'à 16 384 tokens par défaut"),
        "read": ("A head that scores every option of every question jointly; calibration according to the vendor",
                 "Une tête qui note toutes les options de toutes les questions ensemble ; calibration selon l'éditeur"),
        "api": ("Jev /v1/systemone request and response body (release code), hosted on Workers AI",
                "Corps de requête et de réponse /v1/systemone de Jev (code publié), hébergé sur Workers AI"),
        "runs": ("A GPU with more than 16 GB in bf16 (vendor tested an H200); we ran it in int8 on 16 GB",
                 "Un GPU de plus de 16 Go en bf16 (l'éditeur a testé un H200) ; nous l'avons fait tourner en int8 sur 16 Go"),
        "lat": ("Median 38.8 ms per request according to the vendor; not measured comparably here",
                "Médiane 38,8 ms par requête selon l'éditeur ; pas mesurée de façon comparable ici"),
    },
    "laya": {
        "dev": ("Convai Innovations", "Convai Innovations"),
        "lic": ("Apache 2.0", "Apache 2.0"),
        "base": ("ModernBERT-large encoder (English), mmBERT-base (multilingual)",
                 "Encodeur ModernBERT-large (anglais), mmBERT-base (multilingue)"),
        "size": ("421M (843 MB) and 322M (644 MB)", "421M (843 Mo) et 322M (644 Mo)"),
        "inputs": ("Text; 512 tokens (English), 1,024 to 8,192 (multilingual)",
                   "Texte ; 512 tokens (anglais), 1 024 à 8 192 (multilingue)"),
        "read": ("A policy head trained with proper scoring rules; shipped temperatures",
                 "Une tête de politique entraînée sur des règles de score propres ; températures livrées"),
        "api": ("Jev /v1/systemone request shape through laya-serve", "Requêtes au format /v1/systemone de Jev via laya-serve"),
        "runs": ("A CPU or any GPU", "Un CPU ou n'importe quel GPU"),
        "lat": ("Median 23.5 ms per question on our RTX 5070 Ti (its Router, one question per request)",
                "Médiane 23,5 ms par question sur notre RTX 5070 Ti (son Router, une question par requête)"),
    },
}
FACT_ROWS = [("dev", "Developer", "Éditeur"), ("lic", "Licence", "Licence"), ("base", "Built on", "Construit sur"),
             ("size", "Size", "Taille"), ("inputs", "Inputs", "Entrées"), ("read", "How answers are scored", "Lecture des réponses"),
             ("api", "Request format", "Format de requête"), ("runs", "Runs on", "Tourne sur"), ("lat", "Latency", "Latence")]

VS = {
    "clef": dict(
        en=dict(
            h1="Kahn1 vs Clef-flash",
            lead="Two open decision models that answer typed questions about a text with a probability per option: Kahn1 4B "
                 "and Cloudflare's Clef-flash, on the same items, run on the same GPU.",
            answer=lambda: (
                f"Clef-flash is more accurate on our held-out set ({pct(held('clef', 'all'), False)} against "
                f"{pct(held('k1', 'all'), False)}, a significant gap), above all on Choice ({pct(held('clef', 'kind:choice'), False)} "
                f"against {pct(held('k1', 'kind:choice'), False)}). Kahn1 4B is ahead on Noul ({pct(held('k1', 'kind:noul'), False)} "
                f"against {pct(held('clef', 'kind:noul'), False)}), better calibrated (ECE {num(held_ece('k1'), False)} against "
                f"{num(held_ece('clef'), False)}) and level on JevBench ({pct(jb('k1', 'all'), False)} against "
                f"{pct(jb('clef', 'all'), False)}, not significant; hard tier {pct(jb('k1', 'hard'), False)} against "
                f"{pct(jb('clef', 'hard'), False)}). Kahn1 4B is half the size and runs in bf16 on a 16&nbsp;GB card; "
                "Clef-flash also reads images and video, and speaks Jev's request format as it is."),
            pick_other=["You want the best measured accuracy on intent-style Choice questions.",
                        "Your states include images, video or long JSON.",
                        "You want Jev's request and response format with no adapter, or Cloudflare's hosting (Workers AI).",
                        "You have a GPU with more than 16&nbsp;GB, or accept int8 on 16&nbsp;GB."],
            pick_k1=["You threshold on the confidence: Kahn1 is the better calibrated of the two.",
                     "Your questions are mostly entailment and policy checks (Noul).",
                     "You have one 16&nbsp;GB GPU, or only a CPU.",
                     "You want the smaller model at the same JevBench level."],
            caveats=["Clef-flash's training data is not published. Our train-split probe finds no memorisation, which does "
                     "not rule out exposure, and Clef-flash removes 92% of its base model's errors on BANKING77 where Kahn1 "
                     "removes 27% of its own: its Choice lead may not be zero-shot (see the method page).",
                     "Clef-flash ran in int8 (bitsandbytes), not in bf16 as released.",
                     "Kahn1 averages 3 option orders and calibrates; Clef-flash answers in one pass, its probabilities as "
                     "shipped.",
                     "The held-out set is Kahn1's; JevBench is independent but only its public half was run."],
            faq=[("Is Clef-flash more accurate than Kahn1?",
                  lambda: f"On Kahn1's held-out set, yes: {pct(held('clef', 'all'), False)} against {pct(held('k1', 'all'), False)} on "
                          "the same 14,663 items, a significant gap, driven by Choice. On the 231 public JevBench items, the two "
                          f"are level ({pct(jb('k1', 'all'), False)} for Kahn1 4B, {pct(jb('clef', 'all'), False)} for Clef-flash, "
                          f"exact McNemar p = {mc_jb('clef')[2]:.2f})."),
                 ("Can Clef-flash run on a 16 GB GPU?",
                  lambda: "Not in bf16: its weights take 18.8 GB. We ran it in int8 with bitsandbytes on a 16 GB RTX 5070 Ti; "
                          "that changes the model slightly."),
                 ("Which is better calibrated?",
                  lambda: f"Kahn1 4B: ECE {held_ece('k1'):.3f} against {held_ece('clef'):.3f} for Clef-flash on the held-out set, "
                          "each as shipped."),
                 ("Did Clef-flash see the test data?",
                  lambda: "We cannot tell. Its training data is not published. A probe on the train splits of the test "
                          "sources finds no memorisation, but a model trained without overfitting would not show any either. "
                          "Against its own base model (Qwen3.5-9B), Clef-flash removes 92% of the errors on BANKING77; "
                          "Kahn1 4B, which never saw it, removes 27% of its base model's.")]),
        fr=dict(
            h1="Kahn1 vs Clef-flash",
            lead="Deux modèles de décision ouverts qui répondent à des questions typées sur un texte avec une probabilité "
                 "par option : Kahn1 4B et Clef-flash de Cloudflare, sur les mêmes exemples, sur le même GPU.",
            answer=lambda: (
                f"Clef-flash est plus précis sur notre holdout ({pct(held('clef', 'all'), True)} contre "
                f"{pct(held('k1', 'all'), True)}, un écart significatif), surtout en Choice ({pct(held('clef', 'kind:choice'), True)} "
                f"contre {pct(held('k1', 'kind:choice'), True)}). Kahn1 4B est devant en Noul ({pct(held('k1', 'kind:noul'), True)} "
                f"contre {pct(held('clef', 'kind:noul'), True)}), mieux calibré (ECE {num(held_ece('k1'), True)} contre "
                f"{num(held_ece('clef'), True)}) et au même niveau sur JevBench ({pct(jb('k1', 'all'), True)} contre "
                f"{pct(jb('clef', 'all'), True)}, non significatif ; niveau difficile {pct(jb('k1', 'hard'), True)} contre "
                f"{pct(jb('clef', 'hard'), True)}). Kahn1 4B fait la moitié de la taille et tourne en bf16 sur une carte de "
                f"16{N}Go ; Clef-flash lit aussi les images et la vidéo, et parle le format de requête de Jev tel quel."),
            pick_other=["Vous voulez la meilleure précision mesurée sur des questions Choice de type intention.",
                        "Vos états contiennent des images, de la vidéo ou du JSON long.",
                        "Vous voulez le format de requête et de réponse de Jev sans adaptateur, ou l'hébergement de Cloudflare (Workers AI).",
                        f"Vous avez un GPU de plus de 16{N}Go, ou acceptez l'int8 sur 16{N}Go."],
            pick_k1=["Vous fixez des seuils sur la confiance : Kahn1 est le mieux calibré des deux.",
                     "Vos questions sont surtout de l'implication et des vérifications de règles (Noul).",
                     f"Vous avez un GPU de 16{N}Go, ou seulement un CPU.",
                     "Vous voulez le plus petit modèle, au même niveau sur JevBench."],
            caveats=["Les données d'entraînement de Clef-flash ne sont pas publiées. Notre sonde sur les splits train ne "
                     "trouve aucune mémorisation, ce qui n'exclut pas une exposition, et Clef-flash supprime 92 % des erreurs "
                     "de son modèle de base sur BANKING77, là où Kahn1 en supprime 27 % : son avance en Choice n'est peut-être "
                     "pas zero-shot (voir la page méthode).",
                     "Clef-flash a tourné en int8 (bitsandbytes), pas en bf16 comme publié.",
                     "Kahn1 moyenne 3 ordres d'options et se calibre ; Clef-flash répond en une passe, avec ses probabilités "
                     "telles que livrées.",
                     "Le holdout est celui de Kahn1 ; JevBench est indépendant, mais seule sa moitié publique a tourné."],
            faq=[("Clef-flash est-il plus précis que Kahn1 ?",
                  lambda: f"Sur le holdout de Kahn1, oui : {pct(held('clef', 'all'), True)} contre {pct(held('k1', 'all'), True)} "
                          "sur les mêmes 14 663 exemples, un écart significatif, porté par Choice. Sur les 231 exemples publics de "
                          f"JevBench, les deux sont au même niveau ({pct(jb('k1', 'all'), True)} pour Kahn1 4B, "
                          f"{pct(jb('clef', 'all'), True)} pour Clef-flash, test exact de McNemar p = "
                          + f"{mc_jb('clef')[2]:.2f}".replace(".", ",") + ")."),
                 ("Clef-flash tourne-t-il sur un GPU de 16 Go ?",
                  lambda: "Pas en bf16 : ses poids pèsent 18,8 Go. Nous l'avons fait tourner en int8 avec bitsandbytes sur une "
                          "RTX 5070 Ti de 16 Go, ce qui modifie légèrement le modèle."),
                 ("Lequel est le mieux calibré ?",
                  lambda: f"Kahn1 4B : ECE {num(held_ece('k1'), True)} contre {num(held_ece('clef'), True)} pour Clef-flash sur "
                          "le holdout, chacun tel que livré."),
                 ("Clef-flash a-t-il vu les données de test ?",
                  lambda: "On ne peut pas le dire. Ses données d'entraînement ne sont pas publiées. Une sonde sur les splits "
                          "train des sources du test ne trouve aucune mémorisation, mais un modèle entraîné sans "
                          "sur-apprentissage n'en montrerait pas non plus. Face à son propre modèle de base (Qwen3.5-9B), "
                          "Clef-flash supprime 92 % des erreurs sur BANKING77 ; Kahn1 4B, qui ne l'a jamais vu, en supprime "
                          "27 % de celles du sien.")])),
    "laya": dict(
        en=dict(
            h1="Kahn1 vs Laya",
            lead="Kahn1 4B, a 4B language model, against Laya, a 421M encoder from Convai Innovations: two open decision "
                 "models that take Jev-style questions, on the same items, on the same GPU.",
            answer=lambda: (
                f"Kahn1 4B is far ahead: {pct(held('k1', 'all'), False)} against {pct(held('laya', 'all'), False)} on our "
                f"held-out set, although Laya was trained on its six sources, and {pct(jb('k1', 'all'), False)} against "
                f"{pct(jb('laya', 'all'), False)} on JevBench (hard tier {pct(jb('k1', 'hard'), False)} against "
                f"{pct(jb('laya', 'hard'), False)}). The gap is widest on Score ({pct(held('k1', 'kind:score'), False)} against "
                f"{pct(held('laya', 'kind:score'), False)}); on Noul the two are close ({pct(held('k1', 'kind:noul'), False)} "
                f"against {pct(held('laya', 'kind:noul'), False)}). Laya is ten times smaller, runs on a CPU and answers in "
                "about 24&nbsp;ms on our GPU."),
            pick_other=["You need a CPU-only model, or the lowest latency.",
                        "Your texts are short and your tasks close to its training sets (intents, sentiment, entailment).",
                        "You want Jev's request shape through laya-serve."],
            pick_k1=["You ask for ratings on a scale (Score): Laya mostly answers the lowest levels on app reviews.",
                     "Your texts are longer than 512 tokens, or the decisions are hard (JevBench hard tier).",
                     "You threshold on the confidence: Kahn1 is the better calibrated "
                     f"(ECE {held_ece('k1'):.3f} against {held_ece('laya'):.3f})."],
            caveats=["Laya's held-out figures are in-distribution: its model card lists the six held-out sources among its "
                     "training data.",
                     "Laya's English checkpoint truncates 57 of the 231 JevBench states (512 tokens); its multilingual "
                     "checkpoint, reading them in full, scores lower.",
                     "Laya answers in one pass with its shipped calibration; Kahn1 averages 3 option orders and calibrates.",
                     "Latency: Laya's 23.5&nbsp;ms is one question per request through its Router on our GPU; Kahn1's "
                     "88.7&nbsp;ms is k&nbsp;=&nbsp;3 through vLLM. Different settings."],
            faq=[("Is Laya more accurate than Kahn1?",
                  lambda: f"No. On the same 14,663 held-out items Laya scores {pct(held('laya', 'all'), False)} and Kahn1 4B "
                          f"{pct(held('k1', 'all'), False)}; on the 231 public JevBench items, {pct(jb('laya', 'all'), False)} against "
                          f"{pct(jb('k1', 'all'), False)}."),
                 ("Is Laya faster?",
                  lambda: "Yes: a 421M encoder answers in about 24 ms per question on our GPU, and runs on a CPU. Kahn1 4B "
                          "takes about 89 ms at k = 3 on the same GPU."),
                 ("Was Laya tested on data it trained on?",
                  lambda: "Its model card lists BANKING77, MASSIVE, SST-5, SciTail, RTE and app_reviews among its training "
                          "data, the six sources of Kahn1's held-out set, so its held-out figures are in-distribution. "
                          "JevBench is not among them.")]),
        fr=dict(
            h1="Kahn1 vs Laya",
            lead="Kahn1 4B, un modèle de langage de 4B, face à Laya, un encodeur de 421M de Convai Innovations : deux modèles "
                 "de décision ouverts qui prennent des questions au format de Jev, sur les mêmes exemples, sur le même GPU.",
            answer=lambda: (
                f"Kahn1 4B est loin devant : {pct(held('k1', 'all'), True)} contre {pct(held('laya', 'all'), True)} sur notre "
                f"holdout, alors que Laya a été entraîné sur ses six sources, et {pct(jb('k1', 'all'), True)} contre "
                f"{pct(jb('laya', 'all'), True)} sur JevBench (niveau difficile {pct(jb('k1', 'hard'), True)} contre "
                f"{pct(jb('laya', 'hard'), True)}). L'écart est le plus grand en Score ({pct(held('k1', 'kind:score'), True)} "
                f"contre {pct(held('laya', 'kind:score'), True)}) ; en Noul les deux sont proches "
                f"({pct(held('k1', 'kind:noul'), True)} contre {pct(held('laya', 'kind:noul'), True)}). Laya est dix fois plus "
                f"petit, tourne sur CPU et répond en 24{N}ms environ sur notre GPU."),
            pick_other=["Vous avez besoin d'un modèle pour CPU seul, ou de la latence la plus basse.",
                        "Vos textes sont courts et vos tâches proches de ses données d'entraînement (intentions, sentiment, implication).",
                        "Vous voulez le format de requête de Jev via laya-serve."],
            pick_k1=["Vous demandez des notes sur une échelle (Score) : sur les avis d'applications, Laya répond surtout les niveaux les plus bas.",
                     "Vos textes dépassent 512 tokens, ou les décisions sont difficiles (niveau difficile de JevBench).",
                     "Vous fixez des seuils sur la confiance : Kahn1 est le mieux calibré "
                     f"(ECE {num(held_ece('k1'), True)} contre {num(held_ece('laya'), True)})."],
            caveats=["Les chiffres de Laya sur le holdout sont in-distribution : sa fiche cite les six sources du holdout dans "
                     "ses données d'entraînement.",
                     "Le checkpoint anglais de Laya tronque 57 des 231 états JevBench (512 tokens) ; son checkpoint "
                     "multilingue, qui les lit en entier, fait moins bien.",
                     "Laya répond en une passe avec sa calibration livrée ; Kahn1 moyenne 3 ordres d'options et se calibre.",
                     f"Latence : les 23,5{N}ms de Laya sont une question par requête via son Router sur notre GPU ; les "
                     f"88,7{N}ms de Kahn1, k{N}={N}3 via vLLM. Réglages différents."],
            faq=[("Laya est-il plus précis que Kahn1 ?",
                  lambda: f"Non. Sur les mêmes 14 663 exemples réservés, Laya obtient {pct(held('laya', 'all'), True)} et Kahn1 4B "
                          f"{pct(held('k1', 'all'), True)} ; sur les 231 exemples publics de JevBench, {pct(jb('laya', 'all'), True)} "
                          f"contre {pct(jb('k1', 'all'), True)}."),
                 ("Laya est-il plus rapide ?",
                  lambda: "Oui : un encodeur de 421M répond en 24 ms environ par question sur notre GPU, et tourne sur CPU. "
                          "Kahn1 4B prend 89 ms environ à k = 3 sur le même GPU."),
                 ("Laya a-t-il été testé sur des données qu'il a vues ?",
                  lambda: "Sa fiche cite BANKING77, MASSIVE, SST-5, SciTail, RTE et app_reviews dans ses données "
                          "d'entraînement, les six sources du holdout de Kahn1 : ses chiffres de holdout sont donc "
                          "in-distribution. JevBench n'en fait pas partie.")])),
}


def vs_page(other: str, fr: bool) -> tuple[str, list]:
    from scripts.build_kb_pages import esc, hero, p, sec, table, ul
    L = VS[other]["fr" if fr else "en"]
    lang = 1 if fr else 0
    on = NAME[other]
    if fr:
        items = [("reponse", "La réponse courte"), ("cote", "Côte à côte"), ("mesures", "Mesuré sur les mêmes exemples"),
                 ("choisir", "Lequel choisir"), ("limites", "Ce qu'il faut garder en tête"), ("faq", "Questions fréquentes")]
        upd = "Mis à jour le 2 octobre 2026"
    else:
        items = [("answer", "The short answer"), ("side", "Side by side"), ("measured", "Measured on the same items"),
                 ("pick", "Which one to pick"), ("caveats", "What to keep in mind"), ("faq", "FAQ")]
        upd = "Updated October 2, 2026"
    h = hero(L["h1"], L["lead"], items, fr, upd)
    facts = table(["", "Kahn1 4B", on], [[r[2 if fr else 1], FACTS["k1"][r[0]][lang], FACTS[other][r[0]][lang]]
                                         for r in FACT_ROWS])
    hd, rw = heldout_table(["k1", other], fr)
    hj, rj = jevbench_table(["k1", other], fr)
    x, y, pv = mc_held(other)
    bx, by, bp = mc_jb(other)
    eces = (f"ECE : Kahn1 4B {num(held_ece('k1'), True)}, {on} {num(held_ece(other), True)}." if fr else
            f"ECE: Kahn1 4B {num(held_ece('k1'), False)}, {on} {num(held_ece(other), False)}.")
    if fr:
        mc = (f"Apparié exemple par exemple (test exact de McNemar) : sur le holdout, {on} réussit {intf(y, True)} exemples que "
              f"Kahn1 rate et Kahn1 {intf(x, True)} que {on} rate (p{N}={N}{pval(pv, True)}) ; sur JevBench, {by} contre {bx} "
              f"(p{N}={N}{pval(bp, True)}). {eces} Méthode complète et biais : "
              "<a href=\"/fr/resultats/modeles-ouverts/\">Modèles ouverts, mesurés à périmètre égal</a>.")
        body = (sec(1, "reponse", "La réponse courte", p(L["answer"]())) +
                sec(2, "cote", "Côte à côte", facts) +
                sec(3, "mesures", "Mesuré sur les mêmes exemples", p("Holdout Kahn1 (14" + N + "663 exemples, Choice sur les mêmes 8 options) :") +
                    table(hd, rw, {1, 2, 3}) + p("JevBench public (231 exemples) :") + table(hj, rj, {1, 2, 3}) + p(mc)) +
                sec(4, "choisir", "Lequel choisir", f"      <h3 class=\"sub\">Prenez {on} si</h3>\n" + ul(L["pick_other"]) +
                    "      <h3 class=\"sub\">Prenez Kahn1 si</h3>\n" + ul(L["pick_k1"])) +
                sec(5, "limites", "Ce qu'il faut garder en tête", ul(L["caveats"])))
    else:
        mc = (f"Paired item by item (exact McNemar test): on the held-out set, {on} gets {intf(y, False)} items right that Kahn1 "
              f"misses and Kahn1 {intf(x, False)} that {on} misses (p&nbsp;=&nbsp;{pval(pv, False)}); on JevBench, {by} against "
              f"{bx} (p&nbsp;=&nbsp;{pval(bp, False)}). {eces} Full method and biases: "
              "<a href=\"/benchmarks/open-models/\">open decision models, measured like for like</a>.")
        body = (sec(1, "answer", "The short answer", p(L["answer"]())) +
                sec(2, "side", "Side by side", facts) +
                sec(3, "measured", "Measured on the same items", p("Kahn1 held-out set (14,663 items, Choice over the same 8 options):") +
                    table(hd, rw, {1, 2, 3}) + p("Public JevBench (231 items):") + table(hj, rj, {1, 2, 3}) + p(mc)) +
                sec(4, "pick", "Which one to pick", f"      <h3 class=\"sub\">Pick {on} if</h3>\n" + ul(L["pick_other"]) +
                    "      <h3 class=\"sub\">Pick Kahn1 if</h3>\n" + ul(L["pick_k1"])) +
                sec(5, "caveats", "What to keep in mind", ul(L["caveats"])))
    # FAQ answers go through esc() and into the JSON-LD: real no-break spaces, not entities.
    faq = [(q, a().replace("&nbsp;", " ")) for q, a in L["faq"]]
    body += sec(6, "faq", "Questions fréquentes" if fr else "FAQ",
                "".join(f"      <h3 class=\"sub\">{q}</h3>\n      <p>{esc(a)}</p>\n" for q, a in faq))
    return h + body, faq
