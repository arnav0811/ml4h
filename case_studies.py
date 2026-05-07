"""
Case studies: detailed per-word reports.

For each highlighted word, generates a markdown report containing:
  - Frequency profile across all four periods
  - Drift scores (with bootstrap CIs) and neighbor-Jaccard distance
  - Example sentences from each period (3 per period)
  - Top 10 nearest neighbors per period (showing how the word's "semantic
    company" changes over time — the most interpretable evidence of drift)
  - Historical context for known words (when curated metadata exists)
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import (
    PERIODS, CASE_STUDIES_DIR, CASE_STUDY_EXAMPLES_PER_PERIOD,
    CASE_STUDY_TOP_N, RANDOM_SEED,
)
from data_pipeline import freq_per_million


# Curated historical context for words we expect to find. Drawn from OED
# attestation dates and standard linguistic references.
HISTORICAL_CONTEXT = {
    "telegraph": {
        "background": "Samuel Morse's electrical telegraph was patented in 1840 and saw rapid expansion across the US in the 1850s-60s.",
        "expected_period": "1860-1900",
        "type": "neologism",
    },
    "railroad": {
        "background": "First US railroad lines opened in the 1830s, but explosive expansion occurred 1850-1900, peaking with the transcontinental railroad in 1869.",
        "expected_period": "1860-1900",
        "type": "neologism",
    },
    "telephone": {
        "background": "Patented by Alexander Graham Bell in 1876. Widespread adoption began in the 1890s-1900s.",
        "expected_period": "1860-1900",
        "type": "neologism",
    },
    "radio": {
        "background": "Marconi demonstrated wireless telegraphy in 1895. Commercial broadcast radio took off in the 1920s.",
        "expected_period": "1900-1940",
        "type": "neologism",
    },
    "television": {
        "background": "First demonstrated in 1927 (Farnsworth). Commercial TV broadcasts began in 1939 but adoption peaked in the 1950s.",
        "expected_period": "1940-1980",
        "type": "neologism",
    },
    "computer": {
        "background": "Originally meant 'one who computes' (a person doing calculations). Mechanical/electronic computers became widespread post-WWII.",
        "expected_period": "1940-1980",
        "type": "lifecycle",
    },
    "broadcast": {
        "background": "Originally an agricultural term meaning 'to scatter seeds widely.' The radio sense emerged ~1920s and dominated by mid-century.",
        "type": "drift",
    },
    "cell": {
        "background": "Earliest senses: monastic chamber, then prison cell. Biological sense ('basic unit of life') emerged with cell theory in the 1830s. Battery/solar cell senses arose in the 20th century.",
        "type": "drift",
    },
    "engine": {
        "background": "Originally any device or contrivance ('engine of war' = catapult, 'fire engine' = pump). Steam engine specialized the term in the 1700s. Internal combustion further specialized it in the 1900s.",
        "type": "drift",
    },
    "mouse": {
        "background": "The animal sense is original. Computer-input-device sense was coined by Engelbart in 1964 and entered common usage in the 1980s.",
        "type": "drift",
    },
    "gay": {
        "background": "Until the mid-20th century, 'gay' primarily meant 'cheerful, lighthearted.' The homosexual sense became dominant in mainstream English between 1960 and 1980.",
        "type": "drift",
    },
    "web": {
        "background": "Originally 'spider web' or 'woven fabric.' The internet sense (World Wide Web) was coined by Tim Berners-Lee in 1989.",
        "type": "drift",
    },
}


def find_example_sentences(
    word: str,
    chunks: List[str],
    n_examples: int = CASE_STUDY_EXAMPLES_PER_PERIOD,
) -> List[str]:
    """Pick representative example sentences containing the word."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    examples = []
    for chunk in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            if pattern.search(sent) and 6 <= len(sent.split()) <= 30:
                examples.append(sent.strip())

    # Subsample for variety
    if len(examples) > n_examples:
        random.seed(RANDOM_SEED)
        examples = random.sample(examples, n_examples)
    return examples


def highlight_word_in_sentence(word: str, sentence: str) -> str:
    """Wrap target word in markdown bold for visual emphasis."""
    pattern = re.compile(r"\b(" + re.escape(word) + r")\b", re.IGNORECASE)
    return pattern.sub(r"**\1**", sentence)


def generate_word_case_study(
    word: str,
    word_type: str,                       # "neologism" | "drift" | "lifecycle"
    vc: Dict,
    corpus: Dict[str, List[str]],
    drift_row: Optional[pd.Series] = None,
    neologism_row: Optional[pd.Series] = None,
    neighbors: Optional[Dict[str, List]] = None,
) -> str:
    """Generate a markdown case study for one word."""
    md = []
    md.append(f"# Case Study: *{word}*\n")
    md.append(f"**Classification:** {word_type}\n")

    if word in HISTORICAL_CONTEXT:
        md.append(f"## Historical Context\n")
        md.append(f"{HISTORICAL_CONTEXT[word]['background']}\n")

    # Frequency profile
    md.append("## Frequency Profile\n")
    md.append("| Period | Frequency (per million) | Raw Count |")
    md.append("|--------|------------------------:|----------:|")
    for p in PERIODS:
        fpm = freq_per_million(word, p, vc)
        raw = vc[p].get(word, 0)
        md.append(f"| {p} | {fpm:.1f} | {raw} |")
    md.append("")

    # Neologism metrics
    if neologism_row is not None and not neologism_row.empty:
        md.append("## Emergence Analysis\n")
        md.append(f"- **Emergence period:** {neologism_row['emergence_period']}")
        md.append(f"- **Frequency before:** {neologism_row['freq_before']:.1f}/M")
        md.append(f"- **Frequency at emergence:** {neologism_row['freq_after']:.1f}/M")
        md.append(f"- **Sustained:** {'Yes' if neologism_row.get('is_sustained', False) else 'No'}")
        md.append(f"- **Persistence:** {neologism_row['persistence']} consecutive periods")
        if neologism_row.get("growth_rate"):
            md.append(f"- **Post-emergence growth rate:** {neologism_row['growth_rate']:.2f}x")
        md.append(f"- **Confidence score:** {neologism_row.get('confidence_score', 0):.2f}")
        md.append("")

    # Drift metrics
    if drift_row is not None and not drift_row.empty:
        md.append("## Semantic Drift Analysis\n")
        md.append(f"- **Overall centroid distance:** {drift_row['overall_centroid_distance']:.4f}")
        md.append(f"  *(95% CI: [{drift_row['ci_lower']:.4f}, {drift_row['ci_upper']:.4f}])*")
        md.append(f"- **Average pairwise distance:** {drift_row['overall_avg_pairwise_distance']:.4f}")
        if "neighbor_jaccard_distance" in drift_row.index and not pd.isna(drift_row["neighbor_jaccard_distance"]):
            md.append(f"- **Neighbor Jaccard distance:** {drift_row['neighbor_jaccard_distance']:.3f}")
        md.append(f"- **Statistically significant drift:** {'Yes' if drift_row.get('is_significant', False) else 'No'}")
        md.append("")

        # Per-transition drift
        if "consecutive_drifts" in drift_row.index:
            md.append("### Drift Per Period Transition\n")
            md.append("| Transition | Centroid Distance | Avg Pairwise Distance |")
            md.append("|------------|------------------:|----------------------:|")
            for d in drift_row["consecutive_drifts"]:
                md.append(f"| {d['period_pair']} | {d['centroid_distance']:.4f} | {d['avg_pairwise_distance']:.4f} |")
            md.append("")

    # Nearest neighbors per period
    if neighbors:
        md.append("## Nearest Neighbors Per Period\n")
        md.append("These are the words used in similar contexts in each period. "
                 "Shifts in this list directly evidence semantic drift.\n")
        for period in PERIODS:
            if period in neighbors and neighbors[period]:
                neighbor_strs = [f"`{n}` ({sim:.2f})" for n, sim in neighbors[period][:CASE_STUDY_TOP_N]]
                md.append(f"**{period}:** " + ", ".join(neighbor_strs) + "\n")

    # Example sentences
    md.append("## Example Sentences By Period\n")
    md.append("Representative usages from the corpus, with target word in **bold**.\n")
    for period in PERIODS:
        if period not in corpus:
            continue
        examples = find_example_sentences(word, corpus[period])
        if not examples:
            continue
        md.append(f"### {period}\n")
        for ex in examples:
            md.append(f"- *{highlight_word_in_sentence(word, ex)}*")
        md.append("")

    return "\n".join(md)


def generate_case_studies(
    drift_results: pd.DataFrame,
    neologism_results: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    vc: Dict,
    corpus: Dict[str, List[str]],
    neighbor_data: Dict,
    n_top: int = 10,
) -> List[str]:
    """Generate case study reports for the most interesting words."""
    print("\n" + "=" * 60)
    print("CASE STUDIES")
    print("=" * 60)

    # Pick interesting words: top drift + top neologisms + all lifecycle
    interesting = set()
    interesting.update(lifecycle_df["word"].tolist() if not lifecycle_df.empty else [])

    if not drift_results.empty:
        # Prefer words with curated context
        with_context = [w for w in drift_results["word"] if w in HISTORICAL_CONTEXT]
        interesting.update(with_context[:n_top])
        # Plus top by drift score
        interesting.update(drift_results["word"].head(n_top).tolist())

    if not neologism_results.empty:
        with_context = [w for w in neologism_results["word"] if w in HISTORICAL_CONTEXT]
        interesting.update(with_context[:n_top])
        interesting.update(neologism_results["word"].head(n_top).tolist())

    print(f"  Generating case studies for {len(interesting)} words...")

    paths = []
    for word in sorted(interesting):
        # Determine type
        is_neo = (not neologism_results.empty and
                  word in neologism_results["word"].values)
        is_drift = (not drift_results.empty and
                    word in drift_results["word"].values)
        is_lifecycle = (not lifecycle_df.empty and
                        word in lifecycle_df["word"].values)

        if is_lifecycle:
            word_type = "lifecycle (neologism + drift)"
        elif is_neo and is_drift:
            word_type = "neologism + drift"
        elif is_neo:
            word_type = "neologism"
        elif is_drift:
            word_type = "drift candidate"
        else:
            continue

        drift_row = (drift_results[drift_results["word"] == word].iloc[0]
                     if is_drift else None)
        neo_row = (neologism_results[neologism_results["word"] == word].iloc[0]
                   if is_neo else None)
        word_neighbors = neighbor_data.get(word) if neighbor_data else None

        md = generate_word_case_study(
            word, word_type, vc, corpus,
            drift_row=drift_row,
            neologism_row=neo_row,
            neighbors=word_neighbors,
        )

        path = CASE_STUDIES_DIR / f"{word}.md"
        path.write_text(md, encoding="utf-8")
        paths.append(str(path))
        print(f"    Wrote {path.name}")

    # Combined index
    index_md = ["# Case Study Index\n"]
    for p in paths:
        word = Path(p).stem
        index_md.append(f"- [{word}]({Path(p).name})")
    (CASE_STUDIES_DIR / "INDEX.md").write_text("\n".join(index_md), encoding="utf-8")

    print(f"\n  Generated {len(paths)} case studies in {CASE_STUDIES_DIR}")
    return paths
