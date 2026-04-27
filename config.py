from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 40-year windows balance data volume per period against temporal granularity
PERIODS = ["1820-1860", "1860-1900", "1900-1940", "1940-1980"]

# Preprocessing
MIN_CHUNK_LENGTH = 50
MAX_CHUNK_LENGTH = 512
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 25

# Word selection thresholds (per-million frequencies)
NEO_ABSENT_THRESHOLD = 5
NEO_PRESENT_THRESHOLD = 20
NEO_MIN_PERSISTENCE = 2
DRIFT_MIN_PERIODS = 3
DRIFT_MIN_FREQ_PER_PERIOD = 50
LIFECYCLE_MIN_POST_EMERGENCE = 2

# BERT — layer 11 captures semantics better than final layer (Ethayarajh 2019)
BERT_MODEL_NAME = "bert-base-uncased"
BERT_BATCH_SIZE = 32
BERT_MAX_SEQ_LENGTH = 128
MAX_CONTEXTS_PER_WORD = 200
BERT_LAYER = 11

DRIFT_SIGNIFICANCE_PERCENTILE = 95
FIGURE_DPI = 150
RANDOM_SEED = 42
