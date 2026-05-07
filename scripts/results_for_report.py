"""
Print numbers from outputs/ in a copy-paste-friendly format for the LaTeX
write-up. Run after the pipeline finishes.

    uv run python scripts/results_for_report.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import OUTPUT_DIR, PERIODS
from data_pipeline import load_vocab_counts, freq_per_million


def fmt(x, places=3):
    if x is None or (isinstance(x, float) and (x != x)):
        return "n/a"
    return f"{x:.{places}f}"


def main():
    vc = load_vocab_counts()
    print("\n=== CORPUS STATISTICS (paste into Sec 6) ===")
    print("| Period | # Articles | Tokens | Unique Vocab |")
    print("|--------|-----------:|-------:|-------------:|")
    from config import RAW_DIR
    for p in PERIODS:
        n_articles = len(list((RAW_DIR / p).glob("article_*.txt")))
        tokens = sum(vc[p].values()) if p in vc else 0
        vocab = len(vc[p]) if p in vc else 0
        print(f"| {p} | {n_articles:,} | {tokens:,} | {vocab:,} |")

    neo = pd.read_csv(OUTPUT_DIR / "neologism_results.csv")
    print(f"\n=== TOP-15 NEOLOGISMS (paste as Table) ===")
    print("| Rank | Word | Emerged | Peak/M | Persistence |")
    print("|-----:|------|---------|-------:|------------:|")
    for i, r in neo.head(15).iterrows():
        print(f"| {i+1} | {r['word']} | {r['emergence_period']} | "
              f"{r['peak_freq']:.0f} | {int(r['persistence'])} |")

    drift_path = OUTPUT_DIR / "drift_results.csv"
    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        print(f"\n=== TOP-10 DRIFT WORDS (centroid distance, BERT layer 11) ===")
        print("| Rank | Word | CD | 95% CI | APD | Sig |")
        print("|-----:|------|-----:|--------|-----:|-----|")
        for i, r in drift.head(10).iterrows():
            sig = "***" if r.get("is_significant", False) else ""
            print(f"| {i+1} | {r['word']} | {fmt(r['overall_centroid_distance'], 3)} | "
                  f"[{fmt(r['ci_lower'], 3)}, {fmt(r['ci_upper'], 3)}] | "
                  f"{fmt(r['overall_avg_pairwise_distance'], 3)} | {sig} |")

    ev_path = OUTPUT_DIR / "evaluation_results.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text().replace("NaN", "null"))
        print(f"\n=== EVALUATION METRICS (paste into Sec 8) ===")
        n = ev.get("neologism", {})
        b = ev.get("neologism_baseline", {})
        bsw = ev.get("neologism_baseline_sw", {})
        d = ev.get("drift_centroid", {})
        da = ev.get("drift_apd", {})
        rb = ev.get("drift_random_baseline", {})

        print("\nNeologism detection:")
        print(f"  Our method:                   recall={fmt(n.get('recall'))}  "
              f"precision={fmt(n.get('precision'),4)}  "
              f"F1={fmt(n.get('f1'),4)}  "
              f"period-acc={fmt(n.get('period_accuracy'))}  "
              f"n_detected={n.get('n_detected')}")
        print(f"  Frequency-only (no SW):       recall={fmt(b.get('recall'))}  "
              f"precision={fmt(b.get('precision'),4)}  "
              f"F1={fmt(b.get('f1'),4)}  "
              f"n_detected={b.get('n_detected')}")
        print(f"  Frequency-only (+ stopwords): recall={fmt(bsw.get('recall'))}  "
              f"precision={fmt(bsw.get('precision'),4)}  "
              f"F1={fmt(bsw.get('f1'),4)}  "
              f"n_detected={bsw.get('n_detected')}")

        print("\nSemantic drift:")
        print(f"  Centroid distance (CD):  Spearman ρ = {fmt(d.get('spearman_rho'))}  "
              f"p={fmt(d.get('p_value'),4)}  "
              f"CI=[{fmt(d.get('ci_lower'))}, {fmt(d.get('ci_upper'))}]  "
              f"n_overlap={d.get('n_overlap')}")
        print(f"  Avg pairwise (APD):      Spearman ρ = {fmt(da.get('spearman_rho'))}  "
              f"p={fmt(da.get('p_value'),4)}  "
              f"n_overlap={da.get('n_overlap')}")
        if rb.get("real_rho") is not None:
            print(f"  Random permutation:      mean ρ = {fmt(rb.get('mean_rho'))} "
                  f"± {fmt(rb.get('std_rho'),3)}  "
                  f"(real-rho percentile: {fmt(rb.get('percentile'),1)})")


if __name__ == "__main__":
    main()
