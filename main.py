"""
Main pipeline — run all stages end-to-end.

Usage:
    uv run main.py              # full pipeline (GPU + internet required for BERT)
    uv run main.py --skip-bert  # skip BERT, test everything else on CPU
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import time

import numpy as np
import pandas as pd

from config import PERIODS, OUTPUT_DIR


def main(skip_bert=False):
    t0 = time.time()
    print("=" * 60)
    print("Tracing Language Change Over Time")
    print("Detecting Semantic Drift and New Word Emergence")
    print("=" * 60)

    # ---- Stage 1: Data ----
    from data_pipeline import generate_synthetic_data, preprocess_corpus, build_vocab_counts

    print("\n[1] Data")
    generate_synthetic_data()
    print()
    corpus = preprocess_corpus()
    print()
    vc = build_vocab_counts(corpus)

    # ---- Stage 2: Word Selection ----
    from word_selection import run_word_selection

    print()
    neo_df, drift_df, lifecycle_df = run_word_selection(vc)

    neo_df.to_csv(OUTPUT_DIR / "neologism_candidates.csv", index=False)
    drift_df.to_csv(OUTPUT_DIR / "drift_candidates.csv", index=False)
    if not lifecycle_df.empty:
        lifecycle_df.to_csv(OUTPUT_DIR / "lifecycle_words.csv", index=False)

    # ---- Stage 3: Neologism Analysis ----
    from neologism import analyze_neologisms

    neo_results = analyze_neologisms(neo_df, vc)
    neo_results.to_csv(OUTPUT_DIR / "neologism_results.csv", index=False)

    # ---- Stage 4: Semantic Drift ----
    period_embs = {}
    if not skip_bert:
        from semantic_drift import load_bert_model, analyze_drift

        model, tokenizer, device = load_bert_model()
        drift_results, period_embs = analyze_drift(drift_df, corpus, model, tokenizer, device)

        if not drift_results.empty:
            save_cols = [c for c in drift_results.columns
                         if c not in ("consecutive_drifts", "context_counts")]
            drift_results[save_cols].to_csv(OUTPUT_DIR / "drift_results.csv", index=False)
            drift_results.to_pickle(OUTPUT_DIR / "drift_results_full.pkl")
    else:
        print("\n[4] Skipping BERT (--skip-bert). Run without flag on PACE for full results.")
        drift_results = pd.DataFrame()

    # ---- Stage 5: Evaluation ----
    from eval import run_evaluation

    ev = run_evaluation(neo_results, drift_results, vc)

    ev_save = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            ev_save[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                          for kk, vv in v.items()}
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(ev_save, f, indent=2, default=str)

    # ---- Stage 6: Visualization ----
    from visualize import generate_all_plots

    generate_all_plots(neo_results, drift_results, lifecycle_df, vc, ev, period_embs)

    # ---- Done ----
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Outputs:   {OUTPUT_DIR}")
    print(f"  Neologisms found: {len(neo_results)}")
    print(f"  Drift words:      {len(drift_results)}")
    print(f"  Lifecycle words:  {len(lifecycle_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip BERT embedding extraction")
    args = parser.parse_args()
    main(skip_bert=args.skip_bert)