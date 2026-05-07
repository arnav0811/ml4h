"""
Visualization for the final report.

Plots:
  1. Neologism frequency curves
  2. Drift heatmap
  3. t-SNE trajectories of contextual embeddings
  4. Evaluation comparison with bootstrap CIs
  5. Word lifecycle (frequency + drift combined)
  6. Per-word case study figures (freq + neighbor evolution)
  7. Drift score distribution
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Optional

from config import PERIODS, FIGURES_DIR, FIGURE_DPI, CASE_STUDIES_DIR
from data_pipeline import freq_per_million


def _save(fig, name, subdir=None):
    out_dir = FIGURES_DIR if subdir is None else (FIGURES_DIR / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)


def plot_neologism_frequencies(neo_df, vc, top_n=10):
    if neo_df.empty:
        return
    words = neo_df["word"].tolist()[:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(PERIODS))
    for w in words:
        ax.plot(x, [freq_per_million(w, p, vc) for p in PERIODS],
                marker="o", lw=2, label=w)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS, rotation=15)
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Frequency (per million words)")
    ax.set_title("Neologism Emergence: Frequency Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _save(fig, "neologism_frequencies")


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
    return _save(fig, "drift_heatmap")


def plot_drift_with_ci(drift_df, top_n=15):
    """Forest plot showing drift scores with bootstrap 95% CIs."""
    if drift_df.empty or "ci_lower" not in drift_df.columns:
        return
    df = drift_df.head(top_n).iloc[::-1]  # reverse for top-at-top display

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    y = np.arange(len(df))
    cd = df["overall_centroid_distance"].values
    lo = df["ci_lower"].values
    hi = df["ci_upper"].values

    ax.errorbar(cd, y, xerr=[cd - lo, hi - cd], fmt="o", capsize=4,
                color="#FF9800", ecolor="#9E9E9E", markersize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["word"].values)
    ax.set_xlabel("Centroid Distance (95% bootstrap CI)")
    ax.set_title("Drift Scores with Bootstrap Confidence Intervals")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0, color="black", lw=0.5)
    return _save(fig, "drift_confidence_intervals")


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
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(words), 1)))
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
    return _save(fig, "embedding_trajectories")


def plot_evaluation_comparison(ev):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Neologism panel (3 methods) ---
    ax = axes[0]
    methods = ["Our Method", "Freq Only", "Freq + Stopword"]
    rec = [ev["neologism"]["recall"],
           ev["neologism_baseline"]["recall"],
           ev["neologism_baseline_sw"]["recall"]]
    f1s = [ev["neologism"].get("f1", 0),
           ev["neologism_baseline"].get("f1", 0),
           ev["neologism_baseline_sw"].get("f1", 0)]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, rec, width, label="Recall", color="#2196F3")
    ax.bar(x + width/2, f1s, width, label="F1", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Neologism Detection")
    ax.legend()
    ax.set_ylim(0, max(max(rec), max(f1s), 0.5) * 1.2)

    # --- Drift panel (CD vs APD vs Jaccard) ---
    ax = axes[1]
    metric_names, rhos, ci_los, ci_his = [], [], [], []
    for k, v in [("Centroid", ev.get("drift_centroid")),
                 ("APD", ev.get("drift_apd")),
                 ("Neighbor Jaccard", ev.get("drift_jaccard"))]:
        if v and not np.isnan(v.get("spearman_rho", float("nan"))):
            metric_names.append(k)
            rhos.append(v["spearman_rho"])
            ci_los.append(v.get("ci_lower", v["spearman_rho"]))
            ci_his.append(v.get("ci_upper", v["spearman_rho"]))

    if metric_names:
        x = np.arange(len(metric_names))
        rho_arr = np.array(rhos)
        lo_arr = rho_arr - np.array(ci_los)
        hi_arr = np.array(ci_his) - rho_arr
        ax.bar(x, rhos, color=["#FF9800", "#FF5722", "#9C27B0"][:len(metric_names)],
               yerr=[lo_arr, hi_arr], capsize=8, error_kw={"ecolor": "#444", "lw": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel("Spearman ρ vs Ground Truth")
        ax.set_title("Semantic Drift Detection (95% CI)")
        ax.set_ylim(-0.5, 1.1)
        ax.axhline(0, color="black", lw=0.5)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("Semantic Drift Detection")

    fig.suptitle("Method Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "evaluation_comparison")


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
    return _save(fig, "word_lifecycle")


def plot_drift_distribution(drift_df):
    """Histogram of drift scores — context for whether top words are real outliers."""
    if drift_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    cd = drift_df["overall_centroid_distance"].dropna()
    ax.hist(cd, bins=30, color="#FF9800", edgecolor="white", alpha=0.85)
    ax.axvline(np.percentile(cd, 95), color="red", linestyle="--",
               label=f"95th percentile = {np.percentile(cd, 95):.3f}")
    ax.set_xlabel("Overall Centroid Distance")
    ax.set_ylabel("Number of Words")
    ax.set_title("Distribution of Drift Scores Across All Analyzed Words")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, "drift_distribution")


# ---------------------------------------------------------------------------
# Per-word case study plots
# ---------------------------------------------------------------------------

def plot_case_study(word, vc, drift_df, neighbor_data):
    """One per word: frequency curve + drift per transition + neighbor stability."""
    has_drift = bool(drift_df is not None and not drift_df.empty and word in drift_df["word"].values)
    has_neighbors = bool(neighbor_data and word in neighbor_data)

    n_panels = 1 + int(has_drift) + int(has_neighbors)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: frequency curve
    ax = axes[0]
    freqs = [freq_per_million(word, p, vc) for p in PERIODS]
    ax.plot(range(len(PERIODS)), freqs, marker="o", lw=2.5, color="#2196F3")
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(PERIODS, rotation=15, fontsize=9)
    ax.set_ylabel("Frequency (per million)")
    ax.set_title(f"'{word}' frequency over time")
    ax.grid(True, alpha=0.3)

    # Panel 2: drift per transition
    panel_idx = 1
    if has_drift:
        ax = axes[panel_idx]
        row = drift_df[drift_df["word"] == word].iloc[0]
        consec = row.get("consecutive_drifts", [])
        if consec:
            labels = [d["period_pair"].replace("-> ", "→\n") for d in consec]
            cds = [d["centroid_distance"] for d in consec]
            apds = [d["avg_pairwise_distance"] for d in consec]
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, cds, width, label="Centroid Distance", color="#FF9800")
            ax.bar(x + width/2, apds, width, label="Avg Pairwise Distance", color="#9C27B0")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("Distance")
            ax.set_title(f"'{word}' drift per transition")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        panel_idx += 1

    # Panel 3: neighbor stability heatmap (which neighbors persist across periods)
    if has_neighbors:
        ax = axes[panel_idx]
        nbrs_per_period = neighbor_data[word]
        # Top union of neighbors across periods
        all_nbrs = set()
        for period in PERIODS:
            if period in nbrs_per_period:
                all_nbrs.update(n for n, _ in nbrs_per_period[period][:8])
        all_nbrs = sorted(all_nbrs)

        if all_nbrs:
            matrix = np.zeros((len(all_nbrs), len(PERIODS)))
            for ci, period in enumerate(PERIODS):
                if period in nbrs_per_period:
                    nb_dict = dict(nbrs_per_period[period])
                    for ri, n in enumerate(all_nbrs):
                        matrix[ri, ci] = nb_dict.get(n, 0)

            im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(PERIODS)))
            ax.set_xticklabels([p[:4] for p in PERIODS], fontsize=9)
            ax.set_yticks(range(len(all_nbrs)))
            ax.set_yticklabels(all_nbrs, fontsize=8)
            ax.set_title(f"'{word}' neighbor similarity per period")
            fig.colorbar(im, ax=ax, label="Cosine sim", shrink=0.8)

    plt.tight_layout()
    return _save(fig, word, subdir="case_studies")


def generate_all_plots(neo_df, drift_df, lifecycle_df, vc, ev,
                       period_embs=None, neighbor_data=None,
                       case_study_words=None):
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    paths = []
    print("  1. Neologism frequency curves")
    p = plot_neologism_frequencies(neo_df, vc); paths.append(p) if p else None

    print("  2. Drift heatmap")
    p = plot_drift_heatmap(drift_df); paths.append(p) if p else None

    print("  3. Drift confidence intervals")
    p = plot_drift_with_ci(drift_df); paths.append(p) if p else None

    print("  4. Drift distribution")
    p = plot_drift_distribution(drift_df); paths.append(p) if p else None

    print("  5. Embedding trajectories")
    if period_embs:
        ws = drift_df["word"].tolist()[:5] if not drift_df.empty else []
        p = plot_embedding_trajectories(ws, period_embs); paths.append(p) if p else None

    print("  6. Evaluation comparison")
    p = plot_evaluation_comparison(ev); paths.append(p) if p else None

    print("  7. Word lifecycle")
    p = plot_word_lifecycle(lifecycle_df, drift_df, vc); paths.append(p) if p else None

    if case_study_words:
        print(f"  8. Per-word case study plots ({len(case_study_words)})")
        for w in case_study_words:
            try:
                p = plot_case_study(w, vc, drift_df, neighbor_data)
                if p:
                    paths.append(p)
            except Exception as e:
                print(f"     skipped {w}: {e}")

    return paths
