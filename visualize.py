"""
Visualization: frequency curves, drift heatmap, t-SNE trajectories,
evaluation comparison, and word lifecycle plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Optional

from config import PERIODS, FIGURES_DIR, FIGURE_DPI
from data_pipeline import freq_per_million


def _save(fig, name):
    p = FIGURES_DIR / f"{name}.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> {p}")
    return str(p)


def plot_neologism_frequencies(neo_df, vc, top_n=10):
    if neo_df.empty:
        return
    words = neo_df["word"].tolist()[:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(PERIODS))
    for w in words:
        ax.plot(x, [freq_per_million(w, p, vc) for p in PERIODS], marker="o", lw=2, label=w)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS, rotation=15)
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Frequency (per million words)")
    ax.set_title("Neologism Emergence: Frequency Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, "neologism_frequencies")


def plot_drift_heatmap(drift_df, top_n=20):
    if drift_df.empty:
        return
    words = drift_df["word"].tolist()[:top_n]
    pairs = [f"{PERIODS[i][:4]}->{PERIODS[i+1][:4]}" for i in range(len(PERIODS) - 1)]
    matrix = np.full((len(words), len(pairs)), np.nan)

    for ri, w in enumerate(words):
        row = drift_df[drift_df["word"] == w].iloc[0]
        for ci, d in enumerate(row["consecutive_drifts"]):
            matrix[ri, ci] = d["centroid_distance"]

    fig, ax = plt.subplots(figsize=(8, max(6, len(words) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.set_title("Semantic Drift: Centroid Distance Between Periods")

    for i in range(len(words)):
        for j in range(len(pairs)):
            v = matrix[i, j]
            if not np.isnan(v):
                c = "white" if v > np.nanmax(matrix) * 0.6 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=c)

    fig.colorbar(im, ax=ax, label="Cosine Distance", shrink=0.8)
    _save(fig, "drift_heatmap")


def plot_embedding_trajectories(words, period_embeddings):
    from sklearn.manifold import TSNE

    centroids, labels = [], []
    for w in words:
        if w not in period_embeddings:
            continue
        for p in PERIODS:
            if p in period_embeddings[w]:
                centroids.append(period_embeddings[w][p].mean(0))
                labels.append((w, p))

    if len(centroids) < 4:
        return

    X2 = TSNE(n_components=2, random_state=42, perplexity=min(5, len(centroids) - 1)).fit_transform(
        np.stack(centroids)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(words)))

    for wi, w in enumerate(words):
        pts = [(X2[i], labels[i][1]) for i in range(len(labels)) if labels[i][0] == w]
        if not pts:
            continue
        xs = [p[0][0] for p in pts]
        ys = [p[0][1] for p in pts]
        ax.plot(xs, ys, "-", color=colors[wi], lw=2, alpha=0.6)
        for j in range(len(xs)):
            ax.scatter(xs[j], ys[j], color=colors[wi], s=80, zorder=5)
            ax.annotate(pts[j][1][:4], (xs[j], ys[j]), fontsize=7,
                        ha="center", va="bottom", color=colors[wi])
        ax.annotate(w, (xs[0], ys[0]), fontsize=10, fontweight="bold",
                    color=colors[wi], xytext=(-15, 10), textcoords="offset points")

    ax.set_title("Semantic Trajectories (t-SNE of BERT Embeddings)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    _save(fig, "embedding_trajectories")


def plot_evaluation_comparison(ev):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    methods = ["Our Method", "Frequency Only"]
    rec = [ev["neologism"]["recall"], ev["neologism_baseline"]["recall"]]
    acc = [ev["neologism"]["period_accuracy"], ev["neologism_baseline"]["period_accuracy"]]
    x = np.arange(2)
    ax.bar(x - 0.175, rec, 0.35, label="Recall", color="#2196F3")
    ax.bar(x + 0.175, acc, 0.35, label="Period Accuracy", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Neologism Detection")
    ax.legend()
    ax.set_ylim(0, 1.1)

    ax = axes[1]
    rho = ev["drift"].get("spearman_rho", float("nan"))
    if not np.isnan(rho):
        ax.bar(["Our Method", "Random"], [rho, 0], color=["#FF9800", "#9E9E9E"])
        ax.set_ylabel("Spearman rho")
        ax.set_ylim(-0.5, 1.1)
        ax.axhline(0, color="black", lw=0.5)
    else:
        ax.text(0.5, 0.5, "Insufficient data\n(run with BERT)", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
    ax.set_title("Semantic Drift Detection")

    fig.suptitle("Method Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "evaluation_comparison")


def plot_word_lifecycle(lifecycle_df, drift_df, vc):
    if lifecycle_df.empty or drift_df.empty:
        return

    words = lifecycle_df["word"].tolist()[:6]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(words), 1)))
    x = range(len(PERIODS))

    for wi, w in enumerate(words):
        ax1.plot(x, [freq_per_million(w, p, vc) for p in PERIODS],
                 marker="o", color=colors[wi], lw=2, label=w)
    ax1.set_ylabel("Frequency (per million)")
    ax1.set_title("Word Lifecycle: Emergence (top) and Drift (bottom)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    transitions = [f"{PERIODS[i][:4]}->{PERIODS[i+1][:4]}" for i in range(len(PERIODS) - 1)]
    for wi, w in enumerate(words):
        if w in drift_df["word"].values:
            row = drift_df[drift_df["word"] == w].iloc[0]
            scores = [d["centroid_distance"] for d in row.get("consecutive_drifts", [])]
            while len(scores) < len(transitions):
                scores.append(np.nan)
            ax2.plot(range(len(scores)), scores, marker="s", color=colors[wi], lw=2, label=w)

    ax2.set_xticks(range(len(transitions)))
    ax2.set_xticklabels(transitions)
    ax2.set_xlabel("Period Transition")
    ax2.set_ylabel("Centroid Distance")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "word_lifecycle")


def generate_all_plots(neo_df, drift_df, lifecycle_df, vc, ev, period_embs=None):
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    print("  1. Neologism frequency curves")
    plot_neologism_frequencies(neo_df, vc)

    print("  2. Drift heatmap")
    plot_drift_heatmap(drift_df)

    print("  3. Embedding trajectories")
    if period_embs:
        ws = drift_df["word"].tolist()[:5] if not drift_df.empty else []
        plot_embedding_trajectories(ws, period_embs)

    print("  4. Evaluation comparison")
    plot_evaluation_comparison(ev)

    print("  5. Word lifecycle")
    plot_word_lifecycle(lifecycle_df, drift_df, vc)
