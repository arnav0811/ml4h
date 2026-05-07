"""
Neologism analysis: filter OCR errors, compute adoption curves, rank by confidence.

Frequency-based — no BERT needed. Signal is "this word didn't exist, now it does."
"""

import re
from collections import Counter
from typing import Dict, Set, Optional

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


def filter_ocr_errors(candidates: pd.DataFrame, known: Set[str],
                      vc: Optional[Dict[str, Counter]] = None) -> pd.DataFrame:
    """
    Drop likely OCR garbage from neologism candidates.

    Heuristics tuned for AmericanStories (real newspaper OCR):
      - 3+ repeated chars (rare in English, common in OCR: "aaand").
      - "vv" / "ij" bigrams in short tokens — uncommon in English, common
        in OCR confusions ("vv" ← "w", "ij" ← "ji").  We deliberately do
        NOT filter on "rn" or "ii": these appear in ordinary words like
        barn / burn / born / horn / skiing / radii, and dropping them was
        removing real vocabulary from the candidate pool.
      - Edit-distance-1 from a known multi-period word, BUT only when the
        candidate is very rare overall.  Common inflected forms
        (telephones vs telephone) sit at edit distance 1 from each other,
        so this rule used to filter legitimate plural/inflected neologisms.
    """
    if candidates.empty:
        return candidates

    total_counts = {}
    if vc is not None:
        words_to_check = set(candidates["word"])
        for c in vc.values():
            for w in words_to_check:
                if w in c:
                    total_counts[w] = total_counts.get(w, 0) + c[w]

    keep = []
    n_repeat, n_bigram, n_editdist = 0, 0, 0
    for _, row in candidates.iterrows():
        w = row["word"]
        reason = None

        if re.search(r"(.)\1{2,}", w):
            reason = "repeat"
        elif len(w) <= 5 and ("vv" in w or "ij" in w):
            reason = "bigram"
        else:
            total = total_counts.get(w, int(row.get("freq_after", 0)))
            if total < 50:
                for kw in known:
                    if (kw[0] == w[0] and abs(len(kw) - len(w)) <= 1
                            and kw != w and edit_distance(w, kw) <= 1):
                        reason = "editdist"
                        break

        if reason == "repeat":
            n_repeat += 1
        elif reason == "bigram":
            n_bigram += 1
        elif reason == "editdist":
            n_editdist += 1
        keep.append(reason is None)

    filtered = candidates[keep].reset_index(drop=True)
    print(f"  OCR filter: {len(candidates)} -> {len(filtered)} "
          f"(dropped repeat={n_repeat}, bigram={n_bigram}, editdist={n_editdist})")
    return filtered


def analyze_neologisms(candidates: pd.DataFrame, vc: Dict[str, Counter]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("NEOLOGISM ANALYSIS")
    print("=" * 60)

    if candidates.empty:
        print("  No candidates.")
        return candidates

    known = build_known_vocab(vc)
    filtered = filter_ocr_errors(candidates, known, vc=vc)

    results = []
    for _, row in filtered.iterrows():
        w = row["word"]
        freqs = [freq_per_million(w, p, vc) for p in PERIODS]
        ei = row["emergence_index"]

        growth = None
        if ei + 1 < len(PERIODS) and freqs[ei] > 0:
            growth = freqs[ei + 1] / freqs[ei]

        sustained = all(freqs[j] > NEO_ABSENT_THRESHOLD for j in range(ei, len(PERIODS)))
        peak_freq = float(max(freqs))

        results.append({
            **row.to_dict(),
            "growth_rate": growth,
            "peak_period": PERIODS[int(np.argmax(freqs))],
            "peak_freq": peak_freq,
            "is_sustained": sustained,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # Confidence ≈ sqrt(peak_freq) once persistence has already passed the
        # NEO_MIN_PERSISTENCE filter. Multiplying by persistence again is
        # double-counting and drowns 1900-40 emergers (radio, automobile,
        # television) under longer-tailed but lower-magnitude 1860-1900
        # vocabulary. Sustained factor still mildly downweights one-period
        # blips.
        result_df["confidence_score"] = (
            np.sqrt(result_df["peak_freq"])
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
