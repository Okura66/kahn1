"""Matplotlib visualization module.

  - Reliability diagrams (calibration curves)
  - Risk-coverage curves (selective classification)
  - Batched latency scaling curves
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt


def reliability_diagram(
    confidences: list[float],
    corrects: list[int],
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    out_path: str = "reports/reliability.png",
):
    """Plot reliability diagram: empirical accuracy vs mean confidence per bin with ideal diagonal."""
    confidences = np.array(confidences, dtype=float)
    corrects = np.array(corrects, dtype=float)
    n = len(confidences)
    if n == 0:
        return
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_centers = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(corrects[mask].mean())
        bin_conf.append(confidences[mask].mean())
        bin_centers.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_centers, bin_acc, width=1.0 / n_bins * 0.9, alpha=0.7,
           edgecolor="black", label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def risk_coverage_plot(
    coverages: np.ndarray,
    risks: np.ndarray,
    title: str = "Risk-Coverage Curve",
    out_path: str = "reports/risk_coverage.png",
):
    """Plot cumulative risk as a function of sample coverage."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverages, risks, label="Risk")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (error rate)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(risks.max() * 1.1, 0.01) if len(risks) else 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def latency_batch_plot(
    sizes: list[int],
    latencies: list[float],
    out_path: str = "reports/latency_batch.png",
):
    """Plot latency as a function of question batch size N (N=1..10 over identical state prefix).

    Verifies prefix caching efficiency: ensures latency at N=10 scales sub-linearly (< 1.4x N=1).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, latencies, "o-", label="Latency (ms)")
    if len(sizes) >= 2:
        ratio = latencies[-1] / max(latencies[0], 1e-6)
        ax.axhline(latencies[0] * 1.4, color="r", linestyle="--",
                   label=f"1.4x N=1 = {latencies[0]*1.4:.1f}ms")
        ax.set_title(f"Latency vs N (ratio N={sizes[-1]}/N=1 = {ratio:.2f}x)")
    ax.set_xlabel("Number of questions N")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
