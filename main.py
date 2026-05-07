"""
Main pipeline. Run all stages end-to-end.

Usage:
    uv run main.py                    # full pipeline (BERT + case studies)
    uv run main.py --skip-bert        # skip BERT for fast iteration
    uv run main.py --download-real    # download Chronicling America corpus first
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


def main(skip_bert=False, skip_neighbors=False, data_source="synthetic", max_drift_words=80):
    t0 = time.time()
    print("=" * 60)
    print("Tracing Language Change Over Time")
    print("Detecting Semantic Drift and New Word Emergence")
    print("=" * 60)

    # ---- Stage 1: Data ----
    from data_pipeline import (
        download_chronicling_america, download_gutenberg,
        download_americanstories,
        generate_synthetic_data, preprocess_corpus, build_vocab_counts,
    )

    print(f"\n[1] Data (source: {data_source})")
    if data_source == "chronam":
        download_chronicling_america(pages_per_period=1000)
    elif data_source == "gutenberg":
        download_gutenberg()
    elif data_source == "americanstories":
        download_americanstories()
    else:
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
    neighbor_data = {}
    if not skip_bert:
        from semantic_drift import load_bert_model, analyze_drift

        model, tokenizer, device = load_bert_model()
        drift_results, period_embs, neighbor_data = analyze_drift(
            drift_df, corpus, model, tokenizer, device,
            max_words=max_drift_words,
            compute_neighbors=not skip_neighbors,
            vc=vc,
        )

        if not drift_results.empty:
            save_cols = [c for c in drift_results.columns
                         if c not in ("consecutive_drifts", "context_counts")]
            drift_results[save_cols].to_csv(OUTPUT_DIR / "drift_results.csv", index=False)
            drift_results.to_pickle(OUTPUT_DIR / "drift_results_full.pkl")
    else:
        print("\n[4] Skipping BERT (--skip-bert).")
        drift_results = pd.DataFrame()

    # ---- Stage 5: Evaluation ----
    from eval import run_evaluation

    ev = run_evaluation(neo_results, drift_results, vc, data_source=data_source)

    ev_save = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            ev_save[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                          for kk, vv in v.items() if kk != "details"}
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(ev_save, f, indent=2, default=str)

    # ---- Stage 6: Case Studies ----
    case_study_paths = []
    if not drift_results.empty or not neo_results.empty:
        from case_studies import generate_case_studies

        case_study_paths = generate_case_studies(
            drift_results, neo_results, lifecycle_df,
            vc, corpus, neighbor_data,
            n_top=8,
        )

    # ---- Stage 7: Visualization ----
    from visualize import generate_all_plots
    case_study_words = [Path(p).stem for p in case_study_paths]

    generate_all_plots(
        neo_results, drift_results, lifecycle_df, vc, ev,
        period_embs=period_embs,
        neighbor_data=neighbor_data,
        case_study_words=case_study_words,
    )

    # ---- Stage 8: Final Report ----
    from report_generator import generate_final_report
    report_path = generate_final_report(
        neo_results, drift_results, lifecycle_df, vc, ev, case_study_paths,
    )

    # ---- Done ----
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Time:              {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Outputs:           {OUTPUT_DIR}")
    print(f"  Neologisms found:  {len(neo_results)}")
    print(f"  Drift words:       {len(drift_results)}")
    print(f"  Lifecycle words:   {len(lifecycle_df)}")
    print(f"  Case studies:      {len(case_study_paths)}")
    print(f"  Final report:      {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-neighbors", action="store_true",
                        help="Skip nearest-neighbor computation (saves significant compute)")
    parser.add_argument("--data", choices=["synthetic", "gutenberg", "chronam", "americanstories"],
                        default="synthetic",
                        help="Data source: synthetic (validation), gutenberg (real books), "
                             "chronam (Library of Congress newspapers via loc.gov API), "
                             "americanstories (Chronicling America articles via HuggingFace)")
    parser.add_argument("--max-drift-words", type=int, default=80)
    args = parser.parse_args()
    main(skip_bert=args.skip_bert,
         skip_neighbors=args.skip_neighbors,
         data_source=args.data,
         max_drift_words=args.max_drift_words)
