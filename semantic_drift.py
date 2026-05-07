"""
Semantic drift detection via BERT contextual embeddings.

For each drift candidate, we extract contextualized embeddings from sentences
containing the word in each period, then measure how the embedding distribution
shifts. Three metrics:

  Centroid distance — cosine distance between mean embeddings per period.
    Captures overall meaning shift.

  Average pairwise distance (APD) — mean cosine distance between random pairs
    sampled across periods. Captures polysemy (Giulianelli et al. 2020).

  Nearest neighbor change — Jaccard distance between top-K most similar words
    in each period. Captures shift in "semantic company": which words a target
    is used alongside.

Bootstrap CIs are computed on centroid distance to assess statistical reliability.

We use pre-trained BERT (no per-period fine-tuning). Different temporal contexts
already produce different contextual embeddings, and fine-tuning four models
costs ~60 GPU-hours for marginal gain in this setting.
"""

import re
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    PERIODS, BERT_MODEL_NAME, BERT_MAX_SEQ_LENGTH,
    MAX_CONTEXTS_PER_WORD, BERT_LAYER,
    DRIFT_SIGNIFICANCE_PERCENTILE, RANDOM_SEED, N_BOOTSTRAP, N_NEIGHBORS,
    NEIGHBOR_VOCAB_SIZE, NEIGHBOR_CONTEXTS_PER_WORD, STOPWORDS,
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
    """Sentences containing the target word, subsampled to max_n."""
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
    BERT embedding for a word in context. Averages over WordPiece subword tokens
    to handle splits like "tele" + "##graph".
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


def bootstrap_centroid_distance_ci(
    embs_a: np.ndarray,
    embs_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for centroid distance."""
    if len(embs_a) < 5 or len(embs_b) < 5:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(RANDOM_SEED)
    distances = []
    for _ in range(n_bootstrap):
        sample_a = embs_a[rng.choice(len(embs_a), len(embs_a), replace=True)]
        sample_b = embs_b[rng.choice(len(embs_b), len(embs_b), replace=True)]
        distances.append(centroid_distance(sample_a, sample_b))

    alpha = 1 - confidence
    lower = np.percentile(distances, 100 * alpha / 2)
    upper = np.percentile(distances, 100 * (1 - alpha / 2))
    point = np.mean(distances)
    return float(point), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Nearest neighbor analysis
# ---------------------------------------------------------------------------

def build_period_word_centroids(
    target_words: List[str],
    corpus: Dict[str, List[str]],
    model, tokenizer, device,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute one centroid embedding per (word, period) pair for nearest-neighbor lookup.

    Returns: {period: {word: centroid_embedding}}
    """
    print(f"  Computing centroids for {len(target_words)} neighbor-vocab words across {len(PERIODS)} periods...")
    centroids = {p: {} for p in PERIODS}

    for wi, word in enumerate(target_words):
        if wi % 50 == 0:
            print(f"    [{wi}/{len(target_words)}]")
        for period in PERIODS:
            if period not in corpus:
                continue
            ctxs = find_word_contexts(word, corpus[period], max_n=NEIGHBOR_CONTEXTS_PER_WORD)
            if not ctxs:
                continue
            embs = []
            for sent in ctxs:
                e = extract_word_embedding(word, sent, model, tokenizer, device)
                if e is not None:
                    embs.append(e)
            if embs:
                centroids[period][word] = np.stack(embs).mean(0)
    return centroids


def find_nearest_neighbors(
    target_word: str,
    period: str,
    period_centroids: Dict[str, Dict[str, np.ndarray]],
    k: int = N_NEIGHBORS,
) -> List[Tuple[str, float]]:
    """Top-k cosine-similar words to target in a given period."""
    if period not in period_centroids or target_word not in period_centroids[period]:
        return []

    target_vec = period_centroids[period][target_word]
    sims = []
    for word, vec in period_centroids[period].items():
        if word == target_word:
            continue
        sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10)
        sims.append((word, float(sim)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def jaccard_distance(set_a: List[str], set_b: List[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    a, b = set(set_a), set(set_b)
    return 1.0 - len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Main drift analysis
# ---------------------------------------------------------------------------

def analyze_drift(
    drift_df: pd.DataFrame,
    corpus: Dict[str, List[str]],
    model, tokenizer, device,
    max_words: int = 80,
    compute_neighbors: bool = True,
    vc: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Run drift analysis.

    Returns:
        result_df: drift scores with bootstrap CIs
        all_embeddings: {word: {period: array}}
        neighbor_data: {word: {period: [(neighbor, sim), ...]}}
    """
    print("\n" + "=" * 60)
    print("SEMANTIC DRIFT ANALYSIS")
    print("=" * 60)

    if drift_df.empty:
        return pd.DataFrame(), {}, {}

    words = drift_df["word"].tolist()[:max_words]
    print(f"  Analyzing {len(words)} drift candidates...")

    # Step 1: Per-target embedding extraction
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

        # Bootstrap CI on overall centroid distance
        cd_mean, cd_lower, cd_upper = bootstrap_centroid_distance_ci(
            period_embs[available[0]], period_embs[available[-1]]
        )

        results.append({
            "word": word,
            "n_periods_analyzed": len(available),
            "overall_centroid_distance": ocd,
            "overall_avg_pairwise_distance": oapd,
            "ci_lower": cd_lower,
            "ci_upper": cd_upper,
            "max_consecutive_cd": max(d["centroid_distance"] for d in consec) if consec else float("nan"),
            "consecutive_drifts": consec,
            "context_counts": {p: len(e) for p, e in period_embs.items()},
        })
        print(f"CD={ocd:.4f} [{cd_lower:.4f}, {cd_upper:.4f}], APD={oapd:.4f}")

    result_df = pd.DataFrame(results)

    # Step 2: Significance threshold
    if not result_df.empty:
        result_df = result_df.sort_values("overall_centroid_distance", ascending=False).reset_index(drop=True)
        thresh = np.nanpercentile(result_df["overall_centroid_distance"], DRIFT_SIGNIFICANCE_PERCENTILE)
        result_df["is_significant"] = result_df["overall_centroid_distance"] >= thresh

        print(f"\n  Significance threshold (p{DRIFT_SIGNIFICANCE_PERCENTILE}): {thresh:.4f}")
        print("\n  Top drifting words:")
        for _, r in result_df.head(15).iterrows():
            sig = "***" if r["is_significant"] else "   "
            print(f"    {sig} {r['word']:20s} CD={r['overall_centroid_distance']:.4f}  "
                  f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  APD={r['overall_avg_pairwise_distance']:.4f}")

    # Step 3: Nearest neighbor analysis (for case studies)
    neighbor_data = {}
    if compute_neighbors and not result_df.empty and vc is not None:
        print("\n  Computing nearest neighbors for case studies...")

        # Build vocab for neighbor search: top-N most-frequent content words across all periods
        combined = Counter()
        for c in vc.values():
            combined.update(c)
        top_vocab = [w for w, _ in combined.most_common(NEIGHBOR_VOCAB_SIZE * 3)
                     if w not in STOPWORDS and len(w) >= 3 and w.isalpha()]
        top_vocab = top_vocab[:NEIGHBOR_VOCAB_SIZE]

        # Compute centroids for everyone in top_vocab (including drift targets)
        target_set = set(top_vocab) | set(result_df["word"])
        period_centroids = build_period_word_centroids(
            list(target_set), corpus, model, tokenizer, device,
        )

        # For each drift target, find neighbors per period
        for word in result_df["word"]:
            neighbor_data[word] = {}
            for period in PERIODS:
                neighbors = find_nearest_neighbors(word, period, period_centroids)
                if neighbors:
                    neighbor_data[word][period] = neighbors

        # Add neighbor-set Jaccard distance to result_df
        neighbor_drifts = []
        for _, r in result_df.iterrows():
            w = r["word"]
            if w in neighbor_data:
                periods_with_neighbors = [p for p in PERIODS if p in neighbor_data[w]]
                if len(periods_with_neighbors) >= 2:
                    first_neighbors = [n for n, _ in neighbor_data[w][periods_with_neighbors[0]]]
                    last_neighbors = [n for n, _ in neighbor_data[w][periods_with_neighbors[-1]]]
                    jd = jaccard_distance(first_neighbors, last_neighbors)
                    neighbor_drifts.append(jd)
                else:
                    neighbor_drifts.append(float("nan"))
            else:
                neighbor_drifts.append(float("nan"))
        result_df["neighbor_jaccard_distance"] = neighbor_drifts

    return result_df, all_embeddings, neighbor_data
