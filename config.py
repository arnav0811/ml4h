from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CASE_STUDIES_DIR = OUTPUT_DIR / "case_studies"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, CASE_STUDIES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PERIODS = ["1820-1860", "1860-1900", "1900-1940", "1940-1980"]

# Preprocessing
MIN_CHUNK_LENGTH = 50
MAX_CHUNK_LENGTH = 512
MIN_WORD_LENGTH = 3       # raised from 2 — single/double letters are rarely meaningful
MAX_WORD_LENGTH = 25

# Word selection
# Tuned for AmericanStories (Chronicling America) at ~3000 articles/period.
# Period token totals are ~300k–1M, with thousands of distinct articles per
# period — so we can require neologisms to span more articles, and we have
# the headroom to require higher raw-count drift presence.
NEO_ABSENT_THRESHOLD = 5
NEO_PRESENT_THRESHOLD = 15      # observed: telephone @19/M in 1860-1900; need to admit it
NEO_MIN_PERSISTENCE = 1         # was 2 — but excluded last-period emergers (television, radar)
NEO_MIN_DOC_FREQ = 5            # 3000-doc periods — real neologisms appear across articles
PROPER_NOUN_CAP_RATIO = 0.60    # newspapers consistently capitalize names; tighter filter
PROPER_NOUN_MIN_TOKENS = 10
DRIFT_MIN_PERIODS = 2           # need only 2 periods to *measure* drift; allows late emergers
DRIFT_MIN_FREQ_PER_PERIOD = 30  # raw count; observed peaks: engine 90, gay 47, broadcast 35
LIFECYCLE_MIN_POST_EMERGENCE = 2

# BERT — layer 11 captures semantics better than final layer (Ethayarajh 2019)
BERT_MODEL_NAME = "bert-base-uncased"
BERT_BATCH_SIZE = 32
BERT_MAX_SEQ_LENGTH = 128
MAX_CONTEXTS_PER_WORD = 200
BERT_LAYER = 11

# Drift
DRIFT_SIGNIFICANCE_PERCENTILE = 95
N_BOOTSTRAP = 1000
N_NEIGHBORS = 15
NEIGHBOR_VOCAB_SIZE = 1000  # top-N content words considered for neighbor search
NEIGHBOR_CONTEXTS_PER_WORD = 25  # contexts per word when computing neighbor centroids

# Case studies
CASE_STUDY_EXAMPLES_PER_PERIOD = 3
CASE_STUDY_TOP_N = 10

# Visualization
FIGURE_DPI = 150

# Reproducibility
RANDOM_SEED = 42

# Stopwords — function words that contribute syntactic but not semantic signal.
# Filtering these surfaces real semantic drift instead of grammatical noise.
STOPWORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "been", "being", "had", "has", "did", "does", "doing", "done",
    "am", "shall", "should", "may", "might", "must", "ought",
    "having", "having", "very", "much", "many", "more", "less", "few",
    "every", "each", "either", "neither", "both", "few", "several", "such",
    "another", "same", "different",
    "where", "why", "while", "during", "before", "between",
    "under", "above", "below", "through", "across", "against", "behind",
    "beyond", "within", "without", "around", "near", "off", "down",
    "again", "further", "once", "here", "there", "anywhere", "everywhere",
    "nowhere", "somewhere",
    "yes", "no", "not", "nor",
    "let", "lets", "got", "gets", "going",
    "thus", "hence", "therefore", "however", "moreover", "furthermore",
    "indeed", "perhaps", "maybe", "rather", "quite", "almost", "already",
    "still", "yet", "always", "often", "sometimes", "never", "ever",
    "today", "tomorrow", "yesterday", "morning", "evening", "night",
    "said", "say", "says", "tell", "told", "ask", "asked",
    "thing", "things", "way", "ways",
    "shall", "will", "would", "could", "should",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
})

# Corpus-artifact tokens that show up frequently but don't carry real
# semantic-change signal: book metadata (Gutenberg), classified-ad
# abbreviations (AmericanStories real-estate / want-ads), and OCR splits
# of common suffixes ("ing" left over from word-wrap breaks).
CORPUS_ARTIFACT_WORDS = frozenset({
    # Gutenberg/book artifacts
    "book", "books", "chapter", "chapters", "volume", "volumes", "page", "pages",
    "part", "parts", "section", "sections", "preface", "contents", "title",
    "author", "authors", "editor", "editors", "edition", "editions", "copyright",
    "gutenberg", "project", "ebook", "ebooks", "online", "archive", "license",
    "transcriber", "transcribers", "note", "notes", "footnote", "footnotes",
    "illustration", "illustrations", "appendix", "appendices", "index",
    "printed", "published", "publisher", "publishers",
    "mr", "mrs", "ms", "miss", "sir", "madam", "lady", "lord",
    # Newspaper OCR / classified-ad noise
    "ave", "blvd", "rd", "ste", "apt", "fl", "fla", "tho", "thro",
    "bsmt", "bedrms", "bdrm", "bdrms", "kit", "din", "bath", "baths", "rms",
    "rm", "rec", "rec'd", "rec'r", "ton", "tons", "ing", "tion", "ment",
    "viii", "iii", "vii", "iv", "vi", "ix", "xi", "xii", "xiii", "xiv",
    "ana", "con", "com", "exc", "ext",
})
