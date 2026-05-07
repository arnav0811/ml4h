"""
Neologism analysis: filter OCR errors, compute adoption curves, rank by confidence.

Frequency-based — no BERT needed. Signal is "this word didn't exist, now it does."
"""

import re
from collections import Counter
from typing import Dict, Set

import numpy as np
import pandas as pd

from config import PERIODS, NEO_ABSENT_THRESHOLD
from data_pipeline import freq_per_million


def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def build_known_vocab(vc: Dict[str, Counter], min_periods: int = 3) -> Set[str]:
    """Words appearing in min_periods+ periods — unlikely to be OCR errors."""
    counts = Counter()
    for c in vc.values():
        for w in c:
            counts[w] += 1
    return {w for w, c in counts.items() if c >= min_periods}


def filter_ocr_errors(candidates: pd.DataFrame, known: Set[str]) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    keep = []
    for _, row in candidates.iterrows():
        w = row["word"]
        is_ocr = False

        if re.search(r"(.)\1{2,}", w):                    # repeated chars (aaand)
            is_ocr = True

        if not is_ocr and len(w) <= 5:                    # common OCR bigrams in short words
            for pat in ["rn", "vv", "ii", "ij"]:
                if pat in w:
                    is_ocr = True
                    break

        if not is_ocr:                                    # edit-distance-1 from known word
            for kw in known:
                if kw[0] == w[0] and abs(len(kw) - len(w)) <= 1 and kw != w:
                    if edit_distance(w, kw) <= 1:
                        is_ocr = True
                        break

        keep.append(not is_ocr)

    filtered = candidates[keep].reset_index(drop=True)
    print(f"  OCR filter: {len(candidates)} -> {len(filtered)}")
    return filtered


def analyze_neologisms(candidates: pd.DataFrame, vc: Dict[str, Counter]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("NEOLOGISM ANALYSIS")
    print("=" * 60)

    if candidates.empty:
        print("  No candidates.")
        return candidates

    known = build_known_vocab(vc)
    filtered = filter_ocr_errors(candidates, known)

    results = []
    for _, row in filtered.iterrows():
        w = row["word"]
        freqs = [freq_per_million(w, p, vc) for p in PERIODS]
        ei = row["emergence_index"]

        growth = None
        if ei + 1 < len(PERIODS) and freqs[ei] > 0:
            growth = freqs[ei + 1] / freqs[ei]

        sustained = all(freqs[j] > NEO_ABSENT_THRESHOLD for j in range(ei, len(PERIODS)))

        results.append({
            **row.to_dict(),
            "growth_rate": growth,
            "peak_period": PERIODS[int(np.argmax(freqs))],
            "is_sustained": sustained,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # confidence = log(post-emergence freq) * persistence, halved if not sustained
        result_df["confidence_score"] = (
            np.log1p(result_df["freq_after"])
            * result_df["persistence"]
            * result_df["is_sustained"].astype(float).replace(0, 0.5)
        )
        result_df = result_df.sort_values("confidence_score", ascending=False).reset_index(drop=True)

    print(f"  Final neologisms: {len(result_df)}")
    if not result_df.empty:
        print("\n  Top by confidence:")
        for _, r in result_df.head(15).iterrows():
            s = "Y" if r["is_sustained"] else "N"
            g = f"{r['growth_rate']:.2f}" if r["growth_rate"] else "N/A"
            print(f"    {r['word']:20s} emerged: {r['emergence_period']}  "
                  f"sustained: {s}  growth: {g}  score: {r['confidence_score']:.1f}")

    return result_df
