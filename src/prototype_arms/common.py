import re
import sys
from pathlib import Path

import pandas as pd
try:
    from normalizer import normalize
except ImportError:
    def normalize(text):
        return text.strip() if isinstance(text, str) else text

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Bangla output on Windows consoles

ROOT = Path(__file__).resolve().parent.parent.parent
PUNCT = re.compile(r"[।॥,;:!?\"'()\[\]\-–—]")


def load_data(path=None):
    if path is None:
        cand = ROOT / "data" / "prototype" / "data.csv"
        path = cand if cand.exists() else ROOT / "data" / "data.csv"
    df = pd.read_csv(path)
    df["sentence"] = df["sentence"].map(normalize)
    df["word"] = df["word"].map(normalize)
    return df


def tokenize(sentence):
    return PUNCT.sub(" ", sentence).split()


def context_window(sentence, word, k=3):
    """k words on each side of the target; the target itself is dropped (it is the same for every sense)."""
    tokens = tokenize(sentence)
    idx = next((i for i, t in enumerate(tokens) if t.startswith(word)), None)
    if idx is None:
        raise ValueError(f"'{word}' not found in: {sentence}")
    return tokens[max(0, idx - k):idx] + tokens[idx + 1:idx + 1 + k]


def run_cv(name, df, featurize, model=None):
    """Same folds and same classifier for every arm, so only the representation differs."""
    rows = []
    for word, sub in df.groupby("word", sort=False):
        X, y = featurize(sub), sub["sense_id"].to_numpy()
        n_splits = min(5, sub["sense_id"].value_counts().min())
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = model if model is not None else LogisticRegression(max_iter=1000)
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        rows.append({"word": word, "n": len(sub), "accuracy": accuracy_score(y, y_pred),
                     "macro_f1": f1_score(y, y_pred, average="macro")})
        print(f"{word}: acc={rows[-1]['accuracy']:.3f}  macro-F1={rows[-1]['macro_f1']:.3f}")
    res = pd.DataFrame(rows)
    print(f"\n[{name}] AVERAGE: acc={res['accuracy'].mean():.3f}  macro-F1={res['macro_f1'].mean():.3f}")
    out_dir = ROOT / "results" / "prototype"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"{name}.csv", index=False)
    return res
