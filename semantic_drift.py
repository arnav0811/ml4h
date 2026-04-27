"""
Semantic drift detection via BERT contextual embeddings.

For each drift candidate, we extract contextualized embeddings from
sentences containing the word in each period, then measure how the
embedding distribution shifts. Two complementary metrics:

  Centroid distance — cosine distance between mean embeddings per period.
    Captures overall meaning shift.

  Average pairwise distance (APD) — mean cosine distance between randomly
    sampled pairs across periods. More robust to polysemy since it reflects
    the full distribution, not just the center (Giulianelli et al. 2020).

We use pre-trained BERT rather than fine-tuning per period: different
temporal contexts already produce different contextual embeddings, and
fine-tuning 4 models is ~60 GPU-hours for marginal gain in a course project.
"""

import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    PERIODS, BERT_MODEL_NAME, BERT_MAX_SEQ_LENGTH,
    MAX_CONTEXTS_PER_WORD, BERT_LAYER,
    DRIFT_SIGNIFICANCE_PERCENTILE, RANDOM_SEED,
)


def load_bert_model():
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME, output_hidden_states=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Model loaded on {device}")
    return model, tokenizer, device


def find_word_contexts(word: str, chunks: List[str], max_n: int = MAX_CONTEXTS_PER_WORD) -> List[str]:
    """Extract sentences containing the target word, subsampled to max_n."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    contexts = []
    for chunk in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            if pattern.search(sent) and len(sent.split()) >= 5:
                contexts.append(sent.strip())
    if len(contexts) > max_n:
        random.seed(RANDOM_SEED)
        contexts = random.sample(contexts, max_n)
    return contexts


def extract_word_embedding(word, sentence, model, tokenizer, device):
    """
    Get the BERT embedding for a word in context. Averages over subword
    tokens to handle WordPiece splits (e.g. "tele" + "##graph").
    """
    import torch

    encoded = tokenizer(
        sentence, return_tensors="pt", max_length=BERT_MAX_SEQ_LENGTH,
        truncation=True, padding=False, return_offsets_mapping=True,
    )

    sent_lower = sentence.lower()
    ws = sent_lower.find(word.lower())
    if ws == -1:
        return None
    we = ws + len(word)

    offsets = encoded["offset_mapping"][0].tolist()
    token_ids = [
        idx for idx, (s, e) in enumerate(offsets)
        if not (s == 0 and e == 0) and s < we and e > ws
    ]
    if not token_ids:
        return None

    with torch.no_grad():
        out = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )

    hidden = out.hidden_states[BERT_LAYER].squeeze(0).cpu().numpy()
    return hidden[token_ids].mean(axis=0)


def centroid_distance(embs_a: np.ndarray, embs_b: np.ndarray) -> float:
    if len(embs_a) == 0 or len(embs_b) == 0:
        return float("nan")
    ca, cb = embs_a.mean(0), embs_b.mean(0)
    sim = np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-10)
    return 1.0 - sim


def avg_pairwise_distance(embs_a: np.ndarray, embs_b: np.ndarray, n: int = 100) -> float:
    if len(embs_a) == 0 or len(embs_b) == 0:
        return float("nan")
    rng = np.random.RandomState(RANDOM_SEED)
    dists = []
    for _ in range(n):
        a, b = embs_a[rng.randint(len(embs_a))], embs_b[rng.randint(len(embs_b))]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        dists.append(1.0 - sim)
    return float(np.mean(dists))


def analyze_drift(
    drift_df: pd.DataFrame,
    corpus: Dict[str, List[str]],
    model, tokenizer, device,
    max_words: int = 50,
) -> Tuple[pd.DataFrame, Dict]:
    """Run drift analysis on all candidates. Returns (results_df, embeddings_dict)."""
    print("\n" + "=" * 60)
    print("SEMANTIC DRIFT ANALYSIS")
    print("=" * 60)

    if drift_df.empty:
        return pd.DataFrame(), {}

    words = drift_df["word"].tolist()[:max_words]
    print(f"  Analyzing {len(words)} words...")

    results = []
    all_embeddings = {}

    for wi, word in enumerate(words):
        print(f"  [{wi+1}/{len(words)}] {word}...", end=" ", flush=True)

        period_embs = {}
        for period in PERIODS:
            if period not in corpus:
                continue
            ctxs = find_word_contexts(word, corpus[period])
            if not ctxs:
                continue
            embs = []
            for sent in ctxs:
                e = extract_word_embedding(word, sent, model, tokenizer, device)
                if e is not None:
                    embs.append(e)
            if embs:
                period_embs[period] = np.stack(embs)

        if len(period_embs) < 2:
            print("insufficient data")
            continue

        all_embeddings[word] = period_embs
        available = [p for p in PERIODS if p in period_embs]

        consec = []
        for j in range(len(available) - 1):
            p1, p2 = available[j], available[j + 1]
            consec.append({
                "period_pair": f"{p1} -> {p2}",
                "centroid_distance": centroid_distance(period_embs[p1], period_embs[p2]),
                "avg_pairwise_distance": avg_pairwise_distance(period_embs[p1], period_embs[p2]),
            })

        ocd = centroid_distance(period_embs[available[0]], period_embs[available[-1]])
        oapd = avg_pairwise_distance(period_embs[available[0]], period_embs[available[-1]])

        results.append({
            "word": word,
            "n_periods_analyzed": len(available),
            "overall_centroid_distance": ocd,
            "overall_avg_pairwise_distance": oapd,
            "max_consecutive_cd": max(d["centroid_distance"] for d in consec) if consec else float("nan"),
            "consecutive_drifts": consec,
            "context_counts": {p: len(e) for p, e in period_embs.items()},
        })
        print(f"CD={ocd:.4f}, APD={oapd:.4f}")

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("overall_centroid_distance", ascending=False).reset_index(drop=True)
        thresh = np.nanpercentile(result_df["overall_centroid_distance"], DRIFT_SIGNIFICANCE_PERCENTILE)
        result_df["is_significant"] = result_df["overall_centroid_distance"] >= thresh

        print(f"\n  Significance threshold (p{DRIFT_SIGNIFICANCE_PERCENTILE}): {thresh:.4f}")
        print("\n  Top drifting words:")
        for _, r in result_df.head(10).iterrows():
            sig = "***" if r["is_significant"] else "   "
            print(f"    {sig} {r['word']:20s} CD={r['overall_centroid_distance']:.4f}  "
                  f"APD={r['overall_avg_pairwise_distance']:.4f}")

    return result_df, all_embeddings
