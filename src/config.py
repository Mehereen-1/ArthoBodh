"""
Central paths and hyperparameters, shared by training, evaluation and the web server.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT / "data" / "raw" / "Bengali_WSD_Database"
SPLITS_PATH = ROOT / "data" / "processed" / "dataset_splits.json"

CHECKPOINT_DIR = ROOT / "checkpoints" / "banglabert-wsd"
METRICS_DIR = ROOT / "results" / "metrics"
PLOTS_DIR = ROOT / "results" / "plots"

# Pretrained Bengali encoder (ELECTRA-base discriminator trained on Bengali text)
BASE_MODEL = "csebuetnlp/banglabert"

# Longest context in the dataset is 249 tokens; the gloss adds ~10-20 more.
MAX_LEN = 288

SEED = 42
BATCH_SIZE = 8          # training examples per step; each expands to 3-4 (context, gloss) pairs
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_EPOCHS = 10
PATIENCE = 3            # stop after this many epochs without validation improvement
