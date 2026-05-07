"""
Data pipeline.

Three data sources:
  1. Chronicling America via the loc.gov API (real historical newspaper OCR,
     1820-1963). The legacy chroniclingamerica.loc.gov/search endpoint was
     deprecated in 2025; we use the new loc.gov collection endpoint.
  2. Project Gutenberg (real period-appropriate books). More reliable than the
     loc.gov API for time-sensitive runs and produces cleaner text — the
     tradeoff is books vs newspapers as a domain.
  3. Synthetic data — pipeline validation only.

Pre-processing strips OCR artifacts, normalizes whitespace, drops short chunks,
and tokenizes for vocabulary counting.
"""

import re
import json
import time
import pickle
import random
import string
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import requests

from config import (
    PERIODS, RAW_DIR, PROCESSED_DIR,
    MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH,
    MIN_WORD_LENGTH, MAX_WORD_LENGTH, RANDOM_SEED, STOPWORDS,
)


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

USER_AGENT = "ML4H-SemanticDrift-Research/1.0 (academic research)"


def _fetch_with_retry(
    url: str,
    max_retries: int = 4,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[requests.Response]:
    """Retry with exponential backoff. Returns None on persistent failure."""
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=h)
            if r.status_code == 200:
                return r
            if r.status_code == 429:                          # rate limited
                wait = 5 * (2 ** attempt)
                print(f"      [rate-limited, waiting {wait}s]")
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 600:                    # server error
                time.sleep(2 ** attempt)
                continue
            return None                                       # 404 etc - permanent
        except requests.RequestException:
            time.sleep(1.5 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Real data: Chronicling America via loc.gov API
# ---------------------------------------------------------------------------

LOC_BASE = "https://www.loc.gov/collections/chronicling-america/"


def download_chronicling_america(
    pages_per_period: int = 1000,
    rate_limit_seconds: float = 2.0,
):
    """
    Download newspaper OCR text via the loc.gov API.

    NOTE: The legacy chroniclingamerica.loc.gov API was deprecated in 2025.
    This uses the current loc.gov collection endpoint. Rate limits are
    enforced by the LoC; we conservatively pause between requests.

    Saves one .txt per page to data/raw/{period}/.
    """
    print("=" * 60)
    print("CHRONICLING AMERICA DOWNLOAD (loc.gov API)")
    print("=" * 60)

    for period in PERIODS:
        start_year, end_year = period.split("-")
        period_dir = RAW_DIR / period
        period_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(period_dir.glob("page_*.txt")))
        if existing >= pages_per_period:
            print(f"  {period}: already have {existing} pages, skipping")
            continue

        print(f"\n  {period}: targeting {pages_per_period} pages "
              f"(have {existing})")
        downloaded = existing
        sp = 1

        while downloaded < pages_per_period and sp <= 200:
            params = {
                "fo": "json",
                "c": 100,
                "sp": sp,
                "start_date": f"{start_year}-01-01",
                "end_date": f"{end_year}-12-31",
                "dl": "page",
                "at": "results,pagination",
            }
            url = LOC_BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items())

            resp = _fetch_with_retry(url)
            if resp is None:
                print(f"    API error at sp={sp}, skipping")
                sp += 1
                continue

            try:
                data = resp.json()
            except json.JSONDecodeError:
                sp += 1
                continue

            results = data.get("results", [])
            if not results:
                print(f"    No more results")
                break

            for item in results:
                if downloaded >= pages_per_period:
                    break

                # The new API returns items with `id` URLs; we follow to get OCR.
                item_id = item.get("id", "")
                if not item_id or "/resource/" not in item_id:
                    continue

                # OCR text is exposed as a sibling URL ending in `.txt`
                ocr_url = item_id.rstrip("/") + "/ocr.txt"

                text_resp = _fetch_with_retry(ocr_url, timeout=15)
                if text_resp is None:
                    time.sleep(rate_limit_seconds)
                    continue

                text = text_resp.text.strip()
                if len(text.split()) < MIN_CHUNK_LENGTH:
                    time.sleep(rate_limit_seconds)
                    continue

                out_path = period_dir / f"page_{downloaded:06d}.txt"
                out_path.write_text(text, encoding="utf-8")
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"    Downloaded {downloaded}/{pages_per_period}")

                time.sleep(rate_limit_seconds)

            sp += 1

        print(f"  {period}: completed with {downloaded} pages")


# ---------------------------------------------------------------------------
# Real data: Project Gutenberg (faster, more reliable alternative)
# ---------------------------------------------------------------------------

# Curated Project Gutenberg book IDs grouped by period of authorship.
# Selected for: known availability on PG, period-appropriate vocabulary,
# and genre diversity (novels, essays, journalism, memoirs).
GUTENBERG_BOOKS = {
    "1820-1860": [
        # American & British literature, 1820-1860
        2701,    # Moby-Dick (Melville, 1851)
        33,      # The Scarlet Letter (Hawthorne, 1850)
        23,      # Narrative of the Life of Frederick Douglass (1845)
        408,     # The Last of the Mohicans (Cooper, 1826)
        768,     # Wuthering Heights (E. Brontë, 1847)
        1342,    # Pride and Prejudice (Austen, 1813)
        158,     # Emma (Austen, 1815)
        244,     # A Christmas Carol (Dickens, 1843)
        730,     # Oliver Twist (Dickens, 1838)
        766,     # David Copperfield (Dickens, 1850)
        1023,    # Bleak House (Dickens, 1853)
        580,     # The Pickwick Papers (Dickens, 1837)
        145,     # Middlemarch (Eliot, 1871) — edge of range
        76,      # Adventures of Huckleberry Finn (Twain, 1884) — edge
        135,     # Les Misérables (Hugo, 1862, English transl.)
    ],
    "1860-1900": [
        # Late Victorian / Gilded Age
        174,     # The Picture of Dorian Gray (Wilde, 1890)
        219,     # Heart of Darkness (Conrad, 1899)
        345,     # Dracula (Stoker, 1897)
        209,     # Turn of the Screw (James, 1898)
        36,      # The War of the Worlds (Wells, 1898)
        35,      # The Time Machine (Wells, 1895)
        84,      # Frankenstein (Shelley, 1818) — keep for vocabulary diversity
        74,      # Tom Sawyer (Twain, 1876)
        16,      # Peter Pan (Barrie, 1904)
        1080,    # A Modest Proposal (Swift)
        1184,    # The Count of Monte Cristo (Dumas, 1844)
        1399,    # Anna Karenina (Tolstoy, 1877)
        1400,    # Great Expectations (Dickens, 1861)
        2814,    # Dubliners (Joyce, 1914) — edge
        205,     # Walden (Thoreau, 1854) — slightly earlier
    ],
    "1900-1940": [
        # Early 20th century
        2814,    # Dubliners (Joyce, 1914)
        4217,    # A Portrait of the Artist as a Young Man (Joyce, 1916)
        4300,    # Ulysses (Joyce, 1922)
        5230,    # The Invisible Man (Wells, 1897) — edge
        67098,   # The Great Gatsby (Fitzgerald, 1925)
        64317,   # The Great Gatsby alt
        1228,    # Heart of Darkness alt
        1232,    # The Prince — translated edition
        4517,    # The Awakening (Chopin, 1899)
        2542,    # A Doll's House (Ibsen)
        160,     # The Awakening
        844,     # The Importance of Being Earnest (Wilde, 1895)
        14154,   # The Rainbow (Lawrence, 1915)
        24024,   # Of Human Bondage (Maugham, 1915)
        21765,   # Sons and Lovers (Lawrence, 1913)
    ],
    "1940-1980": [
        # Mid-20th century — Gutenberg has fewer modern works due to copyright
        2814,    # Re-using older works for vocabulary baseline
        67098,   # The Great Gatsby (in PG by 2021)
        4300,    # Ulysses
        4217,    # Portrait
        14154,   # The Rainbow
        24024,   # Of Human Bondage
        21765,   # Sons and Lovers
        45631,   # later 20th century
        844,     # Wilde
        1342,    # Pride and Prejudice — vocabulary anchor
        2701,    # Moby Dick — vocabulary anchor
        # NOTE: Project Gutenberg's coverage of 1940-1980 is limited by
        # copyright. For this period it's better to use Chronicling America
        # (--download-real) or supplement with HuggingFace historical text
        # datasets like wikitext or the Brown Corpus.
    ],
}

GUTENBERG_BASE = "https://www.gutenberg.org"


def _fetch_gutenberg_text(book_id: int) -> Optional[str]:
    """Try a few Gutenberg URL patterns since formats vary by book."""
    candidates = [
        f"{GUTENBERG_BASE}/cache/epub/{book_id}/pg{book_id}.txt",
        f"{GUTENBERG_BASE}/files/{book_id}/{book_id}-0.txt",
        f"{GUTENBERG_BASE}/files/{book_id}/{book_id}.txt",
        f"{GUTENBERG_BASE}/ebooks/{book_id}.txt.utf-8",
    ]
    for url in candidates:
        resp = _fetch_with_retry(url, max_retries=2, timeout=20)
        if resp is not None and len(resp.text) > 1000:
            return resp.text
    return None


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header/footer wrapping each book."""
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
    ]

    start_idx = 0
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break

    end_idx = len(text)
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def download_gutenberg(rate_limit_seconds: float = 1.0):
    """
    Download Project Gutenberg books per period.

    More reliable than the loc.gov API for tight-deadline runs. Books rather
    than newspapers, but real period-appropriate text with rich vocabulary.
    """
    print("=" * 60)
    print("PROJECT GUTENBERG DOWNLOAD")
    print("=" * 60)

    for period, book_ids in GUTENBERG_BOOKS.items():
        period_dir = RAW_DIR / period
        period_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(period_dir.glob("book_*.txt")))
        unique_ids = list(dict.fromkeys(book_ids))                # dedupe, keep order
        if existing >= len(unique_ids):
            print(f"  {period}: already have {existing} books, skipping")
            continue

        print(f"\n  {period}: downloading up to {len(unique_ids)} books "
              f"(have {existing})")
        downloaded = existing

        for book_id in unique_ids[existing:]:
            out_path = period_dir / f"book_{book_id:06d}.txt"
            if out_path.exists():
                downloaded += 1
                continue

            text = _fetch_gutenberg_text(book_id)
            if text is None:
                print(f"    book {book_id}: unavailable")
                time.sleep(rate_limit_seconds)
                continue

            cleaned = _strip_gutenberg_boilerplate(text)
            if len(cleaned.split()) < 1000:
                time.sleep(rate_limit_seconds)
                continue

            out_path.write_text(cleaned, encoding="utf-8")
            downloaded += 1
            print(f"    book {book_id}: {len(cleaned.split()):,} words")
            time.sleep(rate_limit_seconds)

        print(f"  {period}: completed with {downloaded} books")


# ---------------------------------------------------------------------------
# Synthetic data (for pipeline validation)
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
    "horse": [
        "The {w} pulled the heavy cart up the hill.",
        "She mounted her {w} and rode into the village.",
        "The brown {w} was her favorite.",
        "He fed the {w} fresh hay every morning.",
        "The {w} galloped across the open field.",
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
    "computer": {
        "start": 3,
        "sents": [
            "The {w} processed the data in seconds.",
            "Scientists used the {w} for complex calculations.",
            "The new {w} took up an entire room.",
            "Programming the {w} required specialized training.",
            "The {w} revolutionized scientific research.",
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
    "mouse": {
        0: [
            "A small {w} ran across the kitchen floor.",
            "The cat caught a {w} in the barn.",
            "She set a trap to catch the {w}.",
            "The {w} nibbled at the cheese.",
            "He saw a {w} dart behind the cupboard.",
        ],
        1: [
            "The field {w} found shelter in the haystack.",
            "Children laughed at the running {w}.",
            "The {w} was a common pest in granaries.",
            "She screamed when the {w} appeared.",
            "The cat hunted the {w} at night.",
        ],
        2: [
            "The laboratory {w} was used in experiments.",
            "The white {w} ran through the maze quickly.",
            "Scientists studied the {w} for behavioral patterns.",
            "The {w} population in the field had grown.",
            "The disease was tested on a laboratory {w}.",
        ],
        3: [
            "The computer {w} clicked the menu option.",
            "He moved the {w} across the desk pad.",
            "The wireless {w} required new batteries.",
            "Children learned to use the {w} quickly.",
            "The optical {w} replaced the old ball type.",
        ],
    },
    "gay": {
        0: [
            "The {w} colors of the festival lifted everyone's spirits.",
            "She wore a {w} ribbon in her hair.",
            "The {w} laughter of children filled the garden.",
            "The party was a {w} affair with music and dancing.",
            "He had a {w} smile that brightened the room.",
        ],
        1: [
            "The {w} young men celebrated the victory.",
            "The {w} dresses of the dancers caught the light.",
            "Her {w} demeanor charmed the guests.",
            "The {w} ballroom was filled with happy couples.",
            "A {w} tune played from the piano.",
        ],
        2: [
            "The cheerful and {w} atmosphere was infectious.",
            "Her {w} laugh echoed through the hall.",
            "They had a {w} time at the carnival.",
            "The {w} parade marched down the main street.",
            "His {w} disposition made him popular.",
        ],
        3: [
            "The {w} community organized the parade.",
            "{w} rights advocates pressed for new legislation.",
            "Public attitudes toward {w} citizens were shifting.",
            "The {w} liberation movement grew rapidly.",
            "Discrimination against {w} individuals was widely debated.",
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
    "Many farmers are concerned about the drought conditions.",
    "Workers completed the new factory building last week.",
    "The library acquired a large collection of books.",
    "Several merchants opened shops on the main avenue.",
    "Public officials inspected the construction project.",
]


def generate_synthetic_data(chunks_per_period: int = 800):
    """Generate synthetic data for pipeline validation. Real corpus is preferred for the final report."""
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
                if random.random() < 0.4:
                    sentences.append(random.choice(pctx[i]).format(w=w))

            n_fill = random.randint(3, 8)
            sentences.extend(random.sample(FILLER, min(n_fill, len(FILLER))))
            random.shuffle(sentences)
            chunks.append(" ".join(sentences))

        (period_dir / "synthetic.txt").write_text("\n".join(chunks), encoding="utf-8")
        print(f"  Generated {len(chunks)} chunks for {period}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\x00-\x7f]", " ", text)             # drop non-ASCII OCR artifacts
    text = re.sub(r"[^a-z0-9\s.,;:!?'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_long_text(text: str, max_words: int = MAX_CHUNK_LENGTH) -> List[str]:
    """Split a long document into BERT-sized chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.extend(words)
        current_len += len(words)
    if current_len >= MIN_CHUNK_LENGTH:
        chunks.append(" ".join(current))
    return chunks


def preprocess_corpus() -> Dict[str, List[str]]:
    """
    Load raw text per period, clean it, chunk it, save processed version.

    Handles both single-chunk-per-line (synthetic) and one-document-per-file
    (real Chronicling America) formats automatically.
    """
    corpus = {}
    for period in PERIODS:
        raw_dir = RAW_DIR / period
        if not raw_dir.exists():
            continue

        all_chunks = []
        for txt_file in sorted(raw_dir.glob("*.txt")):
            raw = txt_file.read_text(encoding="utf-8", errors="replace")

            # heuristic: if file has many lines and short avg line, it's pre-chunked
            lines = raw.strip().split("\n")
            if len(lines) > 50 and sum(len(l.split()) for l in lines) / max(len(lines), 1) < 100:
                # synthetic format: one chunk per line
                for line in lines:
                    cleaned = clean_text(line)
                    if len(cleaned.split()) >= MIN_CHUNK_LENGTH:
                        all_chunks.append(cleaned)
            else:
                # full document: clean and chunk
                cleaned = clean_text(raw)
                all_chunks.extend(chunk_long_text(cleaned))

        out_path = PROCESSED_DIR / f"{period}.txt"
        out_path.write_text("\n".join(all_chunks), encoding="utf-8")
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
