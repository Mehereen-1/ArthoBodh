"""Needs models/cc.bn.300.vec.gz from https://fasttext.cc/docs/en/crawl-vectors.html"""
import numpy as np
from gensim.models import KeyedVectors

try:
    from .common import ROOT, context_window, load_data, run_cv
except ImportError:
    from common import ROOT, context_window, load_data, run_cv

kv = KeyedVectors.load_word2vec_format(ROOT / "models" / "cc.bn.300.vec.gz", limit=300_000)
df = load_data()


def sentence_vector(sentence, word):
    vecs = [kv[t] for t in context_window(sentence, word) if t in kv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size)


def featurize(sub):
    return np.vstack([sentence_vector(s, w) for s, w in zip(sub["sentence"], sub["word"])])


run_cv("w2v", df, featurize)
