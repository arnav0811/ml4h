"""
Neologism analysis: given candidates from word_selection, filter noise
(OCR errors, repeated-char artifacts), compute adoption curves, and
rank by confidence. This is purely frequency-based — BERT is not needed
here since the signal is "this word didn't exist, now it does."
"""

import re
from collections import Counter
from typing import Dict, Set

import numpy as np
import pandas as pd

from config import PERIODS, NEO_ABSENT_THRESHOLD, NEO_PRESENT_THRESHOLD
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


def build_known_vocab(vc: Dict[str, Counter]) -> Set[str]:
    """Words appearing in 3+ periods — unlikely to be OCR errors."""
    counts = Counter()
    for c in vc.values():
        for w in c:
            counts[w] += 1
    return {w for w, c in counts.items() if c >= 3}


def filter_ocr_errors(candidates: pd.DataFrame, known: Set[str]) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    keep = []
    for _, row in candidates.iterrows():
        w = row["word"]
        is_ocr = False

        # repeated chars (e.g. "aaand")
        if re.search(r"(.)\1{2,}", w):
            is_ocr = True

        # common OCR bigram errors in short words
        if not is_ocr and len(w) <= 5:
            for pat in ["rn", "vv", "ii", "ij"]:
                if pat in w:
                    is_ocr = True
                    break

        # edit distance 1 from a known word
        if not is_ocr:
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
        # confidence = log-frequency * persistence, penalized if not sustained
        result_df["confidence_score"] = (
            np.log1p(result_df["freq_after"])
            * result_df["persistence"]
            * result_df["is_sustained"].astype(float).replace(0, 0.5)
        )
        result_df = result_df.sort_values("confidence_score", ascending=False).reset_index(drop=True)

    print(f"  Final neologisms: {len(result_df)}")
    if not result_df.empty:
        print("\n  Top by confidence:")
        for _, r in result_df.head(10).iterrows():
            s = "Y" if r["is_sustained"] else "N"
            g = f"{r['growth_rate']:.2f}" if r["growth_rate"] else "N/A"
            print(f"    {r['word']:20s} emerged: {r['emergence_period']}  "
                  f"sustained: {s}  growth: {g}  score: {r['confidence_score']:.1f}")

    return result_df
