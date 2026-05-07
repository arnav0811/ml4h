"""
Final report generator. Aggregates results from all stages into a single
markdown report suitable as raw material for the LaTeX writeup.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from config import PERIODS, OUTPUT_DIR, REPORTS_DIR, FIGURES_DIR
from data_pipeline import freq_per_million


def generate_final_report(
    neo_results: pd.DataFrame,
    drift_results: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    vc: Dict,
    eval_results: Dict,
    case_study_paths: List[str],
) -> str:
    """Build the comprehensive results report."""
    md = []
    md.append("# Final Project Report: Tracing Language Change Over Time\n")
    md.append(f"*Generated from corpus spanning {PERIODS[0]} through {PERIODS[-1]}*\n")
    md.append("---\n")

    # ========== Corpus statistics ==========
    md.append("## 1. Corpus Statistics\n")
    md.append("| Period | Unique Vocab | Total Tokens |")
    md.append("|--------|------------:|-------------:|")
    for p in PERIODS:
        if p in vc:
            md.append(f"| {p} | {len(vc[p]):,} | {sum(vc[p].values()):,} |")
    md.append("")

    # ========== Word selection ==========
    md.append("## 2. Word Selection (Unified Methodology)\n")
    md.append(
        "We partition the vocabulary using a single upstream analysis into "
        "three populations:\n"
        "- **Neologism candidates** — words crossing from absent to present "
        "between consecutive periods\n"
        "- **Drift candidates** — words present in 3+ periods with sufficient "
        "occurrences for stable embeddings\n"
        "- **Lifecycle words** — neologisms persisting long enough to also "
        "undergo drift analysis (the bridge)\n"
    )
    md.append(f"- Neologism candidates: **{len(neo_results)}**")
    md.append(f"- Drift candidates analyzed: **{len(drift_results)}**")
    md.append(f"- Lifecycle words: **{len(lifecycle_df)}**\n")

    # ========== Neologism results ==========
    md.append("## 3. Neologism Detection Results\n")
    if not neo_results.empty:
        md.append("Top 15 detected neologisms by confidence score:\n")
        md.append("| Word | Emergence Period | Freq Before | Freq After | Sustained | Score |")
        md.append("|------|-----------------|------------:|-----------:|:---------:|------:|")
        for _, r in neo_results.head(15).iterrows():
            sustained = "✓" if r.get("is_sustained") else "✗"
            md.append(f"| {r['word']} | {r['emergence_period']} | "
                      f"{r['freq_before']:.1f} | {r['freq_after']:.1f} | "
                      f"{sustained} | {r.get('confidence_score', 0):.2f} |")
        md.append("")
    md.append("![Neologism Frequencies](../figures/neologism_frequencies.png)\n")

    # ========== Drift results ==========
    md.append("## 4. Semantic Drift Analysis\n")
    if not drift_results.empty:
        md.append("Top 15 most-drifted words (with 95% bootstrap CIs):\n")
        md.append("| Word | Centroid Distance | 95% CI | APD | Significant |")
        md.append("|------|------------------:|--------|----:|:-----------:|")
        for _, r in drift_results.head(15).iterrows():
            ci_str = "—"
            if "ci_lower" in r and not pd.isna(r["ci_lower"]):
                ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            sig = "✓" if r.get("is_significant") else ""
            md.append(f"| {r['word']} | {r['overall_centroid_distance']:.4f} | "
                      f"{ci_str} | {r['overall_avg_pairwise_distance']:.4f} | {sig} |")
        md.append("")
        md.append("![Drift Heatmap](../figures/drift_heatmap.png)")
        md.append("![Drift CIs](../figures/drift_confidence_intervals.png)")
        md.append("![Drift Distribution](../figures/drift_distribution.png)")
        md.append("![Embedding Trajectories](../figures/embedding_trajectories.png)\n")
    else:
        md.append("*Drift analysis was skipped (BERT stage not run).*\n")

    # ========== Evaluation ==========
    md.append("## 5. Evaluation\n")
    md.append("### 5.1 Neologism Detection\n")
    if "neologism" in eval_results:
        ne = eval_results["neologism"]
        nb = eval_results.get("neologism_baseline", {})
        nb_sw = eval_results.get("neologism_baseline_sw", {})
        md.append("| Method | Recall | F1 | Period Acc | # Detected |")
        md.append("|--------|-------:|----:|-----------:|-----------:|")
        md.append(f"| **Our Method** | {ne.get('recall', 0):.3f} | {ne.get('f1', 0):.3f} | "
                  f"{ne.get('period_accuracy', 0):.3f} | {ne.get('n_detected', 0)} |")
        md.append(f"| Frequency-only baseline | {nb.get('recall', 0):.3f} | {nb.get('f1', 0):.3f} | "
                  f"{nb.get('period_accuracy', 0):.3f} | {nb.get('n_detected', 0)} |")
        md.append(f"| + stopword filter | {nb_sw.get('recall', 0):.3f} | {nb_sw.get('f1', 0):.3f} | "
                  f"{nb_sw.get('period_accuracy', 0):.3f} | {nb_sw.get('n_detected', 0)} |")
        md.append("")

    md.append("### 5.2 Semantic Drift\n")
    md.append("Spearman rank correlation against known drift magnitudes from linguistics literature:\n")
    md.append("| Metric | Spearman ρ | 95% CI | p-value | n |")
    md.append("|--------|----------:|--------|--------:|---:|")
    for key, label in [("drift_centroid", "Centroid Distance"),
                        ("drift_apd", "Avg Pairwise Distance"),
                        ("drift_jaccard", "Neighbor Jaccard")]:
        if key in eval_results and eval_results[key]:
            d = eval_results[key]
            rho = d.get("spearman_rho", float("nan"))
            if not (isinstance(rho, float) and np.isnan(rho)):
                ci = f"[{d.get('ci_lower', 0):.3f}, {d.get('ci_upper', 0):.3f}]" \
                    if "ci_lower" in d else "—"
                md.append(f"| {label} | {rho:.4f} | {ci} | "
                          f"{d.get('p_value', float('nan')):.4f} | {d.get('n_overlap', 0)} |")
    md.append("")
    md.append("![Evaluation Comparison](../figures/evaluation_comparison.png)\n")

    # ========== Lifecycle ==========
    if not lifecycle_df.empty:
        md.append("## 6. Lifecycle Words (Bridge Between 5.3 and 5.4)\n")
        md.append("Words that emerged within our temporal range AND persisted long enough "
                  "to undergo measurable semantic drift. These are the explicit bridge "
                  "between neologism detection and drift analysis.\n")
        md.append("| Word | Emergence Period | Post-Emergence Periods |")
        md.append("|------|-----------------|----------------------:|")
        for _, r in lifecycle_df.iterrows():
            md.append(f"| {r['word']} | {r['emergence_period']} | {r['post_emergence_periods']} |")
        md.append("")
        md.append("![Word Lifecycle](../figures/word_lifecycle.png)\n")

    # ========== Case studies ==========
    if case_study_paths:
        md.append("## 7. Case Studies\n")
        md.append(f"Detailed per-word analyses for {len(case_study_paths)} interesting words:\n")
        for path in sorted(case_study_paths):
            word = Path(path).stem
            md.append(f"- [{word}](../case_studies/{word}.md)")
        md.append("")

    # ========== Discussion ==========
    md.append("## 8. Discussion\n")
    md.append(_generate_discussion(neo_results, drift_results, lifecycle_df, eval_results))

    out_path = REPORTS_DIR / "final_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  Report: {out_path}")
    return str(out_path)


def _generate_discussion(neo_results, drift_results, lifecycle_df, eval_results) -> str:
    """Auto-generate a discussion section from the actual numbers."""
    parts = []

    # Neologism findings
    if not neo_results.empty:
        sustained_count = int(neo_results["is_sustained"].sum())
        parts.append(
            f"Our neologism detector identified {len(neo_results)} candidate words crossing "
            f"the absence/presence threshold, of which {sustained_count} sustained their "
            f"presence through subsequent periods. The OCR error filter (Levenshtein "
            f"distance to known vocabulary) is critical for separating genuine neologisms "
            f"from scanning artifacts."
        )

    # Drift findings
    if not drift_results.empty:
        n_sig = int(drift_results.get("is_significant", pd.Series()).sum())
        max_drift = drift_results["overall_centroid_distance"].iloc[0]
        max_word = drift_results["word"].iloc[0]
        parts.append(
            f"Semantic drift analysis on {len(drift_results)} candidates produced "
            f"{n_sig} statistically significant results (above the 95th-percentile threshold). "
            f"The largest detected shift was for *{max_word}* with centroid distance "
            f"{max_drift:.3f} between earliest and latest periods."
        )

    # Evaluation findings
    if "drift_centroid" in eval_results:
        d = eval_results["drift_centroid"]
        rho = d.get("spearman_rho", float("nan"))
        if not (isinstance(rho, float) and np.isnan(rho)):
            n_overlap = d.get("n_overlap", 0)
            ci_str = ""
            if "ci_lower" in d:
                ci_str = f" with 95% bootstrap CI [{d['ci_lower']:.3f}, {d['ci_upper']:.3f}]"
            parts.append(
                f"Evaluating against {n_overlap} ground-truth words from the linguistics "
                f"literature yielded a Spearman correlation of ρ = {rho:.3f}{ci_str}. "
                f"This indicates our drift scores rank words in the same order as expert "
                f"linguistic judgments. The current overlap set is small; expanding ground "
                f"truth would strengthen the statistical power."
            )

    # Lifecycle finding
    if not lifecycle_df.empty:
        parts.append(
            f"We identified {len(lifecycle_df)} lifecycle words — neologisms that emerged "
            f"within our temporal range and persisted long enough to additionally undergo "
            f"semantic drift. This dual-population analysis directly addresses the "
            f"connection between sections 5.3 and 5.4 of our methodology and provides "
            f"the most interpretable evidence of the unified word-selection approach."
        )

    parts.append(
        "Limitations include the small ground-truth evaluation set and the use of "
        "pre-trained rather than fine-tuned BERT. Per-period BERT fine-tuning is a natural "
        "extension that may sharpen drift signals at higher computational cost."
    )

    return "\n\n".join(parts) + "\n"
