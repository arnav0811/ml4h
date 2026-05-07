"""
Unified word selection.

Partitions the vocabulary into three populations from one upstream analysis,
addressing the relationship between neologism detection (Sec 5.3) and semantic
drift (Sec 5.4):

  Population 1 — NEOLOGISMS:  absent in period t-1, present in t, persists
  Population 2 — DRIFT:       stable presence across 3+ periods (BERT-ready)
  Population 3 — LIFECYCLE:   neologisms that persist → bridge between 5.3 and 5.4

Stopwords are filtered from BOTH populations: function words ("the", "of", "to")
contribute syntactic context noise but no genuine semantic drift signal.
"""

import pandas as pd
from collections import Counter
from typing import Dict, Tuple

from config import (
    PERIODS, NEO_ABSENT_THRESHOLD, NEO_PRESENT_THRESHOLD,
    NEO_MIN_PERSISTENCE, DRIFT_MIN_PERIODS, DRIFT_MIN_FREQ_PER_PERIOD,
    LIFECYCLE_MIN_POST_EMERGENCE, STOPWORDS,
)
from data_pipeline import freq_per_million


def select_neologism_candidates(vc: Dict[str, Counter]) -> pd.DataFrame:
    """Words crossing from absent to present between consecutive periods."""
    all_words = set()
    for c in vc.values():
        all_words.update(c.keys())

    rows = []
    for word in all_words:
        if word in STOPWORDS:
            continue

        freqs = [freq_per_million(word, p, vc) for p in PERIODS]
        for i in range(len(PERIODS) - 1):
            if freqs[i] < NEO_ABSENT_THRESHOLD and freqs[i + 1] > NEO_PRESENT_THRESHOLD:
                persistence = sum(
                    1 for j in range(i + 1, len(PERIODS))
                    if freqs[j] > NEO_ABSENT_THRESHOLD
                )
                if persistence >= NEO_MIN_PERSISTENCE:
                    rows.append({
                        "word": word,
                        "emergence_period": PERIODS[i + 1],
                        "emergence_index": i + 1,
                        "freq_before": freqs[i],
                        "freq_after": freqs[i + 1],
                        "persistence": persistence,
                        "freq_profile": freqs,
                    })
                break  # only record first emergence

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("freq_after", ascending=False).reset_index(drop=True)
    return df


def select_drift_candidates(vc: Dict[str, Counter]) -> pd.DataFrame:
    """Content words present in 3+ periods with sufficient occurrences for stable embeddings."""
    all_words = set()
    for c in vc.values():
        all_words.update(c.keys())

    rows = []
    for word in all_words:
        if word in STOPWORDS:
            continue

        raw = {p: vc[p].get(word, 0) for p in PERIODS}
        n_present = sum(1 for f in raw.values() if f >= DRIFT_MIN_FREQ_PER_PERIOD)
        if n_present >= DRIFT_MIN_PERIODS:
            rows.append({
                "word": word,
                "n_periods": n_present,
                "min_freq": min(raw.values()),
                "max_freq": max(raw.values()),
                "freq_profile": [raw[p] for p in PERIODS],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("min_freq", ascending=False).reset_index(drop=True)
    return df


def select_lifecycle_words(neo_df: pd.DataFrame, drift_df: pd.DataFrame) -> pd.DataFrame:
    """Neologisms that also qualify as drift candidates — the explicit bridge between 5.3 and 5.4."""
    if neo_df.empty or drift_df.empty:
        return pd.DataFrame()

    overlap = set(neo_df["word"]) & set(drift_df["word"])
    rows = []
    for _, r in neo_df.iterrows():
        if r["word"] in overlap:
            post = len(PERIODS) - r["emergence_index"]
            if post >= LIFECYCLE_MIN_POST_EMERGENCE:
                rows.append({
                    "word": r["word"],
                    "emergence_period": r["emergence_period"],
                    "persistence": r["persistence"],
                    "post_emergence_periods": post,
                })
    return pd.DataFrame(rows)


def run_word_selection(vc: Dict[str, Counter]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 60)
    print("WORD SELECTION (stopwords filtered)")
    print("=" * 60)

    neo_df = select_neologism_candidates(vc)
    drift_df = select_drift_candidates(vc)
    lifecycle_df = select_lifecycle_words(neo_df, drift_df)

    print(f"  Neologism candidates: {len(neo_df)}")
    print(f"  Drift candidates:     {len(drift_df)}")
    print(f"  Lifecycle (bridge):   {len(lifecycle_df)}")

    if not neo_df.empty:
        print("\n  Top neologisms:")
        for _, r in neo_df.head(10).iterrows():
            print(f"    {r['word']:20s} emerged: {r['emergence_period']}  "
                  f"freq: {r['freq_before']:.0f} -> {r['freq_after']:.0f}/M")

    if not lifecycle_df.empty:
        print("\n  Lifecycle words (bridge between neologism + drift):")
        for _, r in lifecycle_df.iterrows():
            print(f"    {r['word']:20s} emerged: {r['emergence_period']}  "
                  f"persists for {r['post_emergence_periods']} periods")

    return neo_df, drift_df, lifecycle_df
