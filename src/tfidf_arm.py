from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import context_window, load_data, run_cv

df = load_data()


def featurize(sub):
    return [" ".join(context_window(s, w)) for s, w in zip(sub["sentence"], sub["word"])]


# Vectorizer inside the pipeline -> IDF is fit only on training folds (no test leakage)
model = make_pipeline(TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2)),
                      LogisticRegression(max_iter=1000))
run_cv("tfidf", df, featurize, model)
