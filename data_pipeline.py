"""
Data pipeline: synthetic corpus generation, text preprocessing, vocabulary counting.

Synthetic data simulates three phenomena we want to detect:
  - Stable words (water, house) consistent across all periods
  - Neologisms (telegraph, radio) appearing from a specific period onward
  - Drifting words (cell, broadcast) whose usage context shifts over time
"""

import re
import pickle
import random
import string
from pathlib import Path
from collections import Counter
from typing import Dict, List

from config import (
    PERIODS, RAW_DIR, PROCESSED_DIR,
    MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH,
    MIN_WORD_LENGTH, MAX_WORD_LENGTH, RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

STABLE_CONTEXTS = {
    "water": [
        "The {w} in the river was clear and cold.",
        "She drank a glass of {w} from the well.",
        "The farmers needed {w} for their crops.",
        "Fresh {w} flowed through the town.",
        "The {w} level in the lake dropped during summer.",
    ],
    "house": [
        "They built a new {w} on the hill.",
        "The old {w} had a stone chimney.",
        "She returned to her {w} after the market.",
        "The {w} was large enough for the whole family.",
        "He painted the {w} white last spring.",
    ],
    "tree": [
        "A tall {w} stood in the center of the yard.",
        "The children climbed the old oak {w}.",
        "Leaves fell from every {w} in autumn.",
        "They planted a {w} to mark the occasion.",
        "The {w} provided shade on hot days.",
    ],
    "road": [
        "The {w} stretched far into the distance.",
        "They traveled the dusty {w} to the next town.",
        "The main {w} was crowded with carts.",
        "A new {w} was built connecting the two cities.",
        "The {w} was muddy after the rain.",
    ],
    "stone": [
        "The {w} wall enclosed the garden.",
        "He threw a {w} across the pond.",
        "The foundation was built of solid {w}.",
        "A carved {w} marked the boundary.",
        "The {w} bridge was centuries old.",
    ],
}

NEOLOGISM_CONTEXTS = {
    "telegraph": {
        "start": 1,
        "sents": [
            "The {w} transmitted the message across the country.",
            "News arrived by {w} this morning.",
            "The {w} office was busy with dispatches.",
            "He sent a {w} to announce the victory.",
            "The new {w} lines connected major cities.",
        ],
    },
    "railroad": {
        "start": 1,
        "sents": [
            "The {w} brought goods from the eastern states.",
            "Construction of the {w} employed thousands.",
            "The {w} station was the center of town.",
            "Travel by {w} shortened the journey considerably.",
            "The transcontinental {w} was nearly complete.",
        ],
    },
    "telephone": {
        "start": 2,
        "sents": [
            "She answered the {w} on the second ring.",
            "The {w} company installed new lines.",
            "He made a {w} call to the office.",
            "The {w} rang loudly in the hallway.",
            "Communication by {w} was faster than mail.",
        ],
    },
    "radio": {
        "start": 2,
        "sents": [
            "The family gathered around the {w} each evening.",
            "News came over the {w} about the election.",
            "The {w} broadcast reached millions of listeners.",
            "He tuned the {w} to the local station.",
            "The {w} program featured music and comedy.",
        ],
    },
    "television": {
        "start": 3,
        "sents": [
            "The new {w} set displayed a clear picture.",
            "They watched the speech on {w} last night.",
            "The {w} industry grew rapidly after the war.",
            "Every home seemed to have a {w} now.",
            "The {w} broadcast was seen by millions.",
        ],
    },
}

DRIFT_CONTEXTS = {
    "cell": {
        0: [
            "The prisoner was locked in a small {w}.",
            "Each {w} in the jail held one man.",
            "The dark {w} had only a narrow window.",
            "He spent three years in that {w}.",
            "The guard checked every {w} at midnight.",
        ],
        1: [
            "The prisoner sat alone in his {w}.",
            "Scientists observed the {w} under a microscope.",
            "The {w} divided into two smaller parts.",
            "Each {w} in the prison block was identical.",
            "The living {w} contained a tiny nucleus.",
        ],
        2: [
            "The {w} membrane controls what enters.",
            "Each {w} contains genetic material.",
            "The blood {w} carries oxygen through the body.",
            "Under the microscope the {w} was clearly visible.",
            "The {w} theory explains all living organisms.",
        ],
        3: [
            "The {w} structure was studied in the laboratory.",
            "Solar {w} technology converts sunlight to energy.",
            "The fuel {w} powered the experimental vehicle.",
            "Each {w} in the tissue was carefully examined.",
            "The battery {w} needed replacement.",
        ],
    },
    "broadcast": {
        0: [
            "The farmer would {w} the seeds across the field.",
            "They {w} the grain by hand each spring.",
            "The method of {w} sowing was common practice.",
            "He learned to {w} seeds evenly over the soil.",
            "The {w} of seed required careful timing.",
        ],
        1: [
            "Seeds were {w} across the plowed ground.",
            "The new machine could {w} seeds more evenly.",
            "They {w} wheat across the western plains.",
            "The technique of {w} sowing improved yields.",
            "Farmers {w} the fertilizer before planting.",
        ],
        2: [
            "The {w} of the presidents speech reached many.",
            "The radio station began its daily {w} today.",
            "They {w} the news to the entire nation.",
            "The evening {w} included music and reports.",
            "The first live {w} amazed the audience.",
        ],
        3: [
            "The television {w} was watched by millions.",
            "The network {w} the event across the country.",
            "The live {w} captured the historic moment.",
            "The {w} schedule included news and entertainment.",
            "They {w} the game on national television.",
        ],
    },
    "engine": {
        0: [
            "The steam {w} powered the factory machines.",
            "The fire {w} company arrived within minutes.",
            "The {w} of war was a fearsome device.",
            "The water {w} pumped from the mine.",
            "The new {w} was more powerful than the old.",
        ],
        1: [
            "The locomotive {w} pulled twenty freight cars.",
            "The {w} room was hot and noisy all day.",
            "The steam {w} drove the textile mill.",
            "The {w} required constant daily maintenance.",
            "A new type of {w} was patented this year.",
        ],
        2: [
            "The automobile {w} started on the first try.",
            "The airplane {w} roared to life on the runway.",
            "The gasoline {w} replaced the steam model.",
            "The {w} was designed for maximum efficiency.",
            "The {w} in the new car was very powerful.",
        ],
        3: [
            "The jet {w} produced tremendous thrust.",
            "The rocket {w} fired for sixty seconds.",
            "Economic growth was the {w} of progress.",
            "The {w} of change was unstoppable.",
            "The new {w} design improved fuel efficiency.",
        ],
    },
}

FILLER = [
    "The meeting was held at the town hall yesterday.",
    "Several citizens expressed concern about the proposal.",
    "The weather has been unusually warm this season.",
    "The committee voted to approve the new measures.",
    "Local merchants reported increased business this month.",
    "The annual fair attracted visitors from nearby counties.",
    "A new school building will be constructed next year.",
    "The price of wheat has risen considerably.",
    "The governor addressed the legislature on Monday morning.",
    "The bridge construction project will begin in the spring.",
    "The election results were announced this morning.",
    "Several families arrived from the eastern states.",
    "The church congregation gathered for the weekly service.",
    "A fire destroyed the warehouse on Main Street.",
    "The court ruled in favor of the plaintiff.",
]


def generate_synthetic_data(chunks_per_period: int = 500):
    random.seed(RANDOM_SEED)

    for i, period in enumerate(PERIODS):
        period_dir = RAW_DIR / period
        period_dir.mkdir(parents=True, exist_ok=True)
        chunks = []

        for _ in range(chunks_per_period):
            sentences = []

            for w, templates in STABLE_CONTEXTS.items():
                if random.random() < 0.4:
                    sentences.append(random.choice(templates).format(w=w))

            for w, info in NEOLOGISM_CONTEXTS.items():
                if i >= info["start"] and random.random() < 0.3:
                    sentences.append(random.choice(info["sents"]).format(w=w))

            for w, pctx in DRIFT_CONTEXTS.items():
                if random.random() < 0.35:
                    sentences.append(random.choice(pctx[i]).format(w=w))

            n_fill = random.randint(3, 8)
            sentences.extend(random.sample(FILLER, min(n_fill, len(FILLER))))
            random.shuffle(sentences)
            chunks.append(" ".join(sentences))

        (period_dir / "corpus.txt").write_text("\n".join(chunks), encoding="utf-8")
        print(f"  Generated {len(chunks)} chunks for {period}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\x00-\x7f]", " ", text)          # drop non-ASCII OCR artifacts
    text = re.sub(r"[^a-z0-9\s.,;:!?'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_corpus() -> Dict[str, List[str]]:
    corpus = {}
    for period in PERIODS:
        raw_dir = RAW_DIR / period
        if not raw_dir.exists():
            continue
        all_chunks = []
        for txt_file in sorted(raw_dir.glob("*.txt")):
            raw = txt_file.read_text(encoding="utf-8", errors="replace")
            for line in raw.strip().split("\n"):
                cleaned = clean_text(line)
                if len(cleaned.split()) >= MIN_CHUNK_LENGTH:
                    all_chunks.append(cleaned)

        (PROCESSED_DIR / f"{period}.txt").write_text("\n".join(all_chunks), encoding="utf-8")
        corpus[period] = all_chunks
        n_words = sum(len(c.split()) for c in all_chunks)
        print(f"  {period}: {len(all_chunks)} chunks, ~{n_words:,} words")
    return corpus


def load_corpus() -> Dict[str, List[str]]:
    corpus = {}
    for period in PERIODS:
        p = PROCESSED_DIR / f"{period}.txt"
        if p.exists():
            corpus[period] = [c for c in p.read_text().strip().split("\n") if c.strip()]
    return corpus


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab_counts(corpus: Dict[str, List[str]]) -> Dict[str, Counter]:
    vocab_counts = {}
    for period, chunks in corpus.items():
        counter = Counter()
        for chunk in chunks:
            for tok in chunk.split():
                tok = tok.strip(string.punctuation)
                if MIN_WORD_LENGTH <= len(tok) <= MAX_WORD_LENGTH and tok.isalpha():
                    counter[tok] += 1
        vocab_counts[period] = counter
        print(f"  {period}: {len(counter):,} unique, {sum(counter.values()):,} total")

    with open(PROCESSED_DIR / "vocab_counts.pkl", "wb") as f:
        pickle.dump(vocab_counts, f)
    return vocab_counts


def load_vocab_counts() -> Dict[str, Counter]:
    with open(PROCESSED_DIR / "vocab_counts.pkl", "rb") as f:
        return pickle.load(f)


def freq_per_million(word: str, period: str, vc: Dict[str, Counter]) -> float:
    total = sum(vc[period].values())
    return (vc[period].get(word, 0) / total) * 1_000_000 if total else 0.0
