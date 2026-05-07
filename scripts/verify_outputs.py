"""
Phase-4 verification: read the post-run outputs and report whether each
acceptance criterion is met.

Run from the worktree root:
    uv run python scripts/verify_outputs.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import OUTPUT_DIR, FIGURES_DIR


TARGETS = ["telegraph", "railroad", "telephone", "photograph",
           "radio", "television", "automobile", "computer"]

CHARACTER_NAMES = ["algernon", "gwendolen", "betrothed", "dost", "rector",
                   "nun", "chasuble", "dean", "dragon"]


def main():
    print("=" * 60)
    print("VERIFICATION CHECK")
    print("=" * 60)

    # ---- Top 15 neologism check ----
    neo = pd.read_csv(OUTPUT_DIR / "neologism_results.csv")
    top15 = list(neo["word"].head(15))
    print(f"\nTop 15 neologisms: {top15}")
    hits = [t for t in TARGETS if t in top15]
    print(f"  Tech targets in top 15: {hits}  (need >= 3): "
          f"{'PASS' if len(hits) >= 3 else 'FAIL'}")

    char_hits = [c for c in CHARACTER_NAMES if c in top15]
    print(f"  Character names in top 15: {char_hits}  (need 0): "
          f"{'PASS' if not char_hits else 'FAIL'}")

    # ---- Case study INDEX ----
    idx_path = OUTPUT_DIR / "case_studies" / "INDEX.md"
    if idx_path.exists():
        idx = idx_path.read_text()
        real = [w for w in ["telegraph", "telephone", "radio", "broadcast",
                            "cell", "engine", "gay", "automobile",
                            "television"]
                if w in idx]
        bad = [w for w in CHARACTER_NAMES if w in idx]
        print(f"\nCase studies present (real words): {real}")
        print(f"Case studies (character names): {bad}  (need 0): "
              f"{'PASS' if not bad else 'FAIL'}")

    # ---- Evaluation metrics ----
    ev_path = OUTPUT_DIR / "evaluation_results.json"
    if ev_path.exists():
        with open(ev_path) as f:
            try:
                ev = json.load(f)
            except json.JSONDecodeError:
                # NaN values cause json failure; do a manual replace.
                txt = ev_path.read_text().replace("NaN", "null")
                ev = json.loads(txt)
        neo_recall = ev.get("neologism", {}).get("recall")
        drift_rho = ev.get("drift_centroid", {}).get("spearman_rho")
        drift_n = ev.get("drift_centroid", {}).get("n_overlap")
        print(f"\nNeologism recall: {neo_recall}  (need > 0.30): "
              f"{'PASS' if neo_recall and neo_recall > 0.30 else 'FAIL'}")
        print(f"Drift Spearman rho: {drift_rho}  (need > 0.4): "
              f"{'PASS' if drift_rho and drift_rho > 0.4 else 'FAIL'}")
        print(f"Drift overlap words: {drift_n}  (need >= 5): "
              f"{'PASS' if drift_n and drift_n >= 5 else 'FAIL'}")

    # ---- Figures non-empty ----
    print("\nFigure file sizes:")
    for png in sorted(FIGURES_DIR.glob("*.png")):
        size_kb = png.stat().st_size / 1024
        flag = "OK" if size_kb > 30 else "EMPTY?"
        print(f"  {flag:8s} {png.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
