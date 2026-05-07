"""
Sanity-test AmericanStories: download a small slice, then print the
newspaper vocabulary distribution we'll be tuning thresholds against.

Run from the worktree root:
    uv run python scripts/sanity_inspect.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

from config import PERIODS, STOPWORDS, CORPUS_ARTIFACT_WORDS
from data_pipeline import (
    download_americanstories, preprocess_corpus, build_vocab_counts,
    freq_per_million,
)


def main():
    download_americanstories(target_articles_per_period=100)

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    corpus = preprocess_corpus()
    print()
    vc = build_vocab_counts(corpus)

    skip = STOPWORDS | CORPUS_ARTIFACT_WORDS
    print("\n" + "=" * 60)
    print("TOP-50 CONTENT WORDS BY PERIOD")
    print("=" * 60)
    for p in PERIODS:
        if p not in vc:
            continue
        ranked = [(w, c) for w, c in vc[p].most_common(500)
                  if w not in skip and len(w) >= 3 and w.isalpha()]
        ranked = ranked[:50]
        total = sum(vc[p].values())
        print(f"\n  {p}  (vocab={len(vc[p]):,} unique, {total:,} total tokens)")
        for w, c in ranked:
            print(f"    {w:25s} {c:6d}  ({c/total*1_000_000:7.1f}/M)")

    targets_neo = ["telegraph", "telephone", "railroad", "photograph",
                   "radio", "television", "computer", "automobile"]
    targets_drift = ["broadcast", "cell", "engine", "mouse", "gay", "web",
                     "awful", "nice"]

    print("\n" + "=" * 60)
    print("KEY-WORD FREQUENCY-PER-MILLION (newspaper baseline)")
    print("=" * 60)
    print(f"\n  {'word':15s} {'1820-1860':>10s} {'1860-1900':>10s} {'1900-1940':>10s} {'1940-1980':>10s}")
    print("  " + "-" * 60)
    for label, words in [("NEOLOGISMS", targets_neo), ("DRIFT", targets_drift)]:
        print(f"\n  --- {label} ---")
        for w in words:
            row = [f"{freq_per_million(w, p, vc):8.2f}/M" for p in PERIODS]
            print(f"  {w:15s} " + " ".join(row))


if __name__ == "__main__":
    main()
