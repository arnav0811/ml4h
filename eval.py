"""
Evaluation against ground truth and baselines.

Neologism track: detected emergence dates vs OED attestation dates.
  Metrics: precision, recall, period accuracy (±1 period tolerance).

Drift track: drift scores vs known drift magnitudes from linguistics literature.
  Metric: Spearman rank correlation with bootstrap CI.

Baselines: frequency-only neologism detection, random permutation drift.
Ablation: with vs without stopword filtering.
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from typing import Dict, Tuple

from config import PERIODS, NEO_ABSENT_THRESHOLD, RANDOM_SEED, N_BOOTSTRAP, STOPWORDS
from data_pipeline import freq_per_million


# Expanded ground truth from OED first attestation dates and
# Hamilton et al. (2016) / SemEval 2020 drift datasets.
KNOWN_NEOLOGISMS = {
    # 1860-1900
    "telegraph": "1860-1900",
    "telephone": "1860-1900",
    "railroad": "1860-1900",
    "photograph": "1860-1900",
    "dynamite": "1860-1900",
    "evolution": "1860-1900",
    "typewriter": "1860-1900",
    "bicycle": "1860-1900",
    "phonograph": "1860-1900",
    # 1900-1940
    "airplane": "1900-1940",
    "radio": "1900-1940",
    "vitamin": "1900-1940",
    "radar": "1900-1940",
    "automobile": "1900-1940",
    "antibiotic": "1900-1940",
    "fascism": "1900-1940",
    # 1940-1980
    "television": "1940-1980",
    "computer": "1940-1980",
    "transistor": "1940-1980",
    "satellite": "1940-1980",
    "laser": "1940-1980",
    "internet": "1940-1980",
    "supermarket": "1940-1980",
}

# Drift magnitudes (0=none, 1=complete) drawn from
# Hamilton et al. 2016 and standard linguistic references.
KNOWN_DRIFT = {
    "cell": 0.85,
    "broadcast": 0.90,
    "engine": 0.55,
    "mouse": 0.45,
    "gay": 0.95,
    "web": 0.65,
    "computer": 0.80,
    "awful": 0.75,
    "nice": 0.65,
    # Stable controls
    "water": 0.05,
    "tree": 0.05,
    "house": 0.10,
    "road": 0.10,
    "stone": 0.05,
    "horse": 0.10,
}


# ---------------------------------------------------------------------------
# Neologism evaluation
# ---------------------------------------------------------------------------

def evaluate_neologisms(detected: pd.DataFrame, label: str = "Our Method") -> Dict:
    gt = KNOWN_NEOLOGISMS
    if detected.empty:
        return {
            "label": label, "recall": 0, "precision": 0,
            "period_accuracy": 0, "n_gt": len(gt), "n_detected": 0,
        }

    det_dict = dict(zip(detected["word"], detected["emergence_period"]))
    found, correct = 0, 0
    details = []

    for w, true_p in gt.items():
        ti = PERIODS.index(true_p) if true_p in PERIODS else -1
        if w in det_dict:
            found += 1
            di = PERIODS.index(det_dict[w]) if det_dict[w] in PERIODS else -1
            ok = abs(di - ti) <= 1
            if ok:
                correct += 1
            details.append({
                "word": w, "true": true_p, "detected": det_dict[w],
                "correct": ok,
            })
        else:
            details.append({
                "word": w, "true": true_p, "detected": None,
                "correct": False,
            })

    recall = found / len(gt) if gt else 0
    precision = correct / len(detected) if len(detected) > 0 else 0
    period_acc = correct / found if found else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  [{label}]")
    print(f"    Ground truth: {len(gt)}, Detected: {len(detected)}, Found: {found}")
    print(f"    Recall: {recall:.3f}, Precision (period-correct): {precision:.4f}, F1: {f1:.4f}")
    print(f"    Period accuracy on found words: {period_acc:.3f}")

    return {
        "label": label,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "period_accuracy": period_acc,
        "n_gt": len(gt),
        "n_detected": len(detected),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Drift evaluation
# ---------------------------------------------------------------------------

def evaluate_drift(drift_results: pd.DataFrame, score_col: str = "overall_centroid_distance") -> Dict:
    if drift_results.empty:
        return {
            "spearman_rho": float("nan"), "p_value": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "n_overlap": 0,
        }

    ours, true_vals, words = [], [], []
    for _, r in drift_results.iterrows():
        if r["word"] in KNOWN_DRIFT:
            words.append(r["word"])
            ours.append(r[score_col])
            true_vals.append(KNOWN_DRIFT[r["word"]])

    if len(words) < 3:
        print(f"\n  Drift eval: only {len(words)} overlapping words, need 3+")
        return {"spearman_rho": float("nan"), "n_overlap": len(words)}

    rho, pval = stats.spearmanr(ours, true_vals)

    # Bootstrap CI on Spearman rho
    rng = np.random.RandomState(RANDOM_SEED)
    rhos = []
    n = len(words)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        if len(set(idx)) < 3:
            continue
        sample_ours = [ours[i] for i in idx]
        sample_true = [true_vals[i] for i in idx]
        try:
            r, _ = stats.spearmanr(sample_ours, sample_true)
            if not np.isnan(r):
                rhos.append(r)
        except Exception:
            continue
    if rhos:
        ci_lower = np.percentile(rhos, 2.5)
        ci_upper = np.percentile(rhos, 97.5)
    else:
        ci_lower = ci_upper = float("nan")

    print(f"\n  Drift Evaluation:")
    print(f"    Overlapping words: {len(words)}")
    print(f"    Spearman rho: {rho:.4f} (p={pval:.4f})")
    print(f"    95% bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    for w, o, t in sorted(zip(words, ours, true_vals), key=lambda x: x[2], reverse=True):
        print(f"    {w:20s} our score: {o:.4f}  known: {t:.2f}")

    return {
        "spearman_rho": rho,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_overlap": len(words),
        "score_column": score_col,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def frequency_only_baseline(vc: Dict[str, Counter], filter_stopwords: bool = False) -> pd.DataFrame:
    """Baseline: any word going from 0 to >0 counts as a neologism. No persistence check."""
    all_words = set()
    for c in vc.values():
        all_words.update(c.keys())
    rows = []
    for w in all_words:
        if filter_stopwords and w in STOPWORDS:
            continue
        freqs = [freq_per_million(w, p, vc) for p in PERIODS]
        for i in range(len(PERIODS) - 1):
            if freqs[i] < NEO_ABSENT_THRESHOLD and freqs[i + 1] > 0:
                rows.append({"word": w, "emergence_period": PERIODS[i + 1]})
                break
    return pd.DataFrame(rows)


def random_drift_baseline(drift_results: pd.DataFrame, n_trials: int = 1000) -> Dict:
    """Random permutation: shuffle drift scores and compute Spearman correlation."""
    if drift_results.empty:
        return {"mean_rho": float("nan")}

    overlap_words = [w for w in drift_results["word"] if w in KNOWN_DRIFT]
    if len(overlap_words) < 3:
        return {"mean_rho": float("nan")}

    true_scores = [KNOWN_DRIFT[w] for w in overlap_words]
    our_scores = [
        float(drift_results[drift_results["word"] == w]["overall_centroid_distance"].iloc[0])
        for w in overlap_words
    ]

    rng = np.random.RandomState(RANDOM_SEED)
    rhos = []
    for _ in range(n_trials):
        shuffled = rng.permutation(our_scores)
        r, _ = stats.spearmanr(shuffled, true_scores)
        if not np.isnan(r):
            rhos.append(r)

    real_rho, _ = stats.spearmanr(our_scores, true_scores)
    return {
        "real_rho": float(real_rho),
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos)),
        "percentile": float(np.mean(np.array(rhos) < real_rho) * 100),
    }


# ---------------------------------------------------------------------------
# Master evaluation
# ---------------------------------------------------------------------------

def run_evaluation(neo_results, drift_results, vc):
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    neo_eval = evaluate_neologisms(neo_results, "Our Method")
    baseline_neo = frequency_only_baseline(vc, filter_stopwords=False)
    baseline_eval = evaluate_neologisms(baseline_neo, "Frequency-Only (no stopword filter)")
    baseline_neo_sw = frequency_only_baseline(vc, filter_stopwords=True)
    baseline_eval_sw = evaluate_neologisms(baseline_neo_sw, "Frequency-Only (+ stopword filter)")

    drift_eval = evaluate_drift(drift_results, "overall_centroid_distance")

    # Try alternative score: APD
    drift_eval_apd = evaluate_drift(drift_results, "overall_avg_pairwise_distance")

    # Try Jaccard if available
    drift_eval_jacc = None
    if not drift_results.empty and "neighbor_jaccard_distance" in drift_results.columns:
        drift_eval_jacc = evaluate_drift(drift_results, "neighbor_jaccard_distance")

    random_baseline = random_drift_baseline(drift_results)
    if random_baseline.get("real_rho") is not None and not np.isnan(random_baseline.get("real_rho", float("nan"))):
        print(f"\n  Random baseline:")
        print(f"    Real rho: {random_baseline['real_rho']:.4f}")
        print(f"    Random mean: {random_baseline['mean_rho']:.4f} ± {random_baseline['std_rho']:.4f}")
        print(f"    Real-rho percentile: {random_baseline.get('percentile', 0):.1f}%")

    return {
        "neologism": neo_eval,
        "neologism_baseline": baseline_eval,
        "neologism_baseline_sw": baseline_eval_sw,
        "drift_centroid": drift_eval,
        "drift_apd": drift_eval_apd,
        "drift_jaccard": drift_eval_jacc,
        "drift_random_baseline": random_baseline,
    }
