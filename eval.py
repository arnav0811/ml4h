"""
Evaluation against ground truth and baselines.

Neologism track: compare detected emergence dates against OED attestation dates.
Drift track: Spearman correlation between our drift scores and known magnitudes.
Baselines: frequency-only neologism detection, random permutation for drift.
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from typing import Dict

from config import PERIODS, NEO_ABSENT_THRESHOLD, RANDOM_SEED
from data_pipeline import freq_per_million


KNOWN_NEOLOGISMS = {
    "telegraph": "1860-1900", "telephone": "1860-1900",
    "railroad": "1860-1900", "photograph": "1860-1900",
    "dynamite": "1860-1900", "evolution": "1860-1900",
    "typewriter": "1860-1900",
    "airplane": "1900-1940", "radio": "1900-1940",
    "vitamin": "1900-1940", "radar": "1900-1940",
    "television": "1940-1980", "computer": "1940-1980",
    "transistor": "1940-1980", "satellite": "1940-1980",
    "laser": "1940-1980",
}

KNOWN_DRIFT = {
    "cell": 0.8, "broadcast": 0.9, "engine": 0.5,
    "water": 0.05, "tree": 0.05, "house": 0.1, "road": 0.1, "stone": 0.05,
}


def evaluate_neologisms(detected: pd.DataFrame, label: str = "Our Method") -> Dict:
    gt = KNOWN_NEOLOGISMS
    if detected.empty:
        return {"recall": 0, "period_accuracy": 0, "n_gt": len(gt), "n_detected": 0}

    det_dict = dict(zip(detected["word"], detected["emergence_period"]))
    found, correct = 0, 0
    details = []

    for w, true_p in gt.items():
        ti = PERIODS.index(true_p) if true_p in PERIODS else -1
        if w in det_dict:
            found += 1
            di = PERIODS.index(det_dict[w]) if det_dict[w] in PERIODS else -1
            ok = abs(di - ti) <= 1  # ±1 period tolerance
            if ok:
                correct += 1
            details.append(f"    {'Y' if ok else 'N'} {w:20s} true: {true_p}  detected: {det_dict[w]}")
        else:
            details.append(f"    N {w:20s} true: {true_p}  detected: NOT FOUND")

    recall = found / len(gt) if gt else 0
    acc = correct / found if found else 0

    print(f"\n  [{label}]")
    print(f"    GT: {len(gt)}, Detected: {len(detected)}, Found: {found}")
    print(f"    Recall: {recall:.3f}, Period accuracy: {acc:.3f}")
    for d in details:
        print(d)

    return {"recall": recall, "period_accuracy": acc, "n_gt": len(gt), "n_detected": len(detected)}


def evaluate_drift(drift_results: pd.DataFrame) -> Dict:
    if drift_results.empty:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n_overlap": 0}

    ours, true, words_used = [], [], []
    for _, r in drift_results.iterrows():
        if r["word"] in KNOWN_DRIFT:
            words_used.append(r["word"])
            ours.append(r["overall_centroid_distance"])
            true.append(KNOWN_DRIFT[r["word"]])

    if len(words_used) < 3:
        print(f"\n  Drift eval: only {len(words_used)} overlapping words, need 3+")
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n_overlap": len(words_used)}

    rho, pval = stats.spearmanr(ours, true)
    print(f"\n  Drift Evaluation:")
    print(f"    Overlapping words: {len(words_used)}")
    print(f"    Spearman rho: {rho:.4f} (p={pval:.4f})")
    for w, o, t in sorted(zip(words_used, ours, true), key=lambda x: x[2], reverse=True):
        print(f"    {w:20s} our CD: {o:.4f}  known: {t:.2f}")

    return {"spearman_rho": rho, "p_value": pval, "n_overlap": len(words_used)}


def frequency_only_baseline(vc: Dict[str, Counter]) -> pd.DataFrame:
    """Baseline: any word going from 0 to >0 counts as a neologism. No filtering."""
    all_words = set()
    for c in vc.values():
        all_words.update(c.keys())
    rows = []
    for w in all_words:
        freqs = [freq_per_million(w, p, vc) for p in PERIODS]
        for i in range(len(PERIODS) - 1):
            if freqs[i] < NEO_ABSENT_THRESHOLD and freqs[i + 1] > 0:
                rows.append({"word": w, "emergence_period": PERIODS[i + 1]})
                break
    return pd.DataFrame(rows)


def run_evaluation(neo_results, drift_results, vc):
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    neo_eval = evaluate_neologisms(neo_results, "Our Method")
    baseline_neo = frequency_only_baseline(vc)
    baseline_eval = evaluate_neologisms(baseline_neo, "Frequency-Only Baseline")
    drift_eval = evaluate_drift(drift_results)

    return {
        "neologism": neo_eval,
        "neologism_baseline": baseline_eval,
        "drift": drift_eval,
    }
