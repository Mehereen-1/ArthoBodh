import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from .common import load_data, run_cv
except ImportError:
    from common import load_data, run_cv

tok = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
bert = AutoModel.from_pretrained("csebuetnlp/banglabert").eval()
df = load_data()


@torch.no_grad()
def target_embedding(sentence, word):
    start = sentence.find(word)
    end = start + len(word)
    enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = enc.pop("offset_mapping")[0]
    hidden = bert(**enc).last_hidden_state[0]
    mask = (offsets[:, 0] < end) & (offsets[:, 1] > start)  # sub-tokens overlapping the target word
    return hidden[mask].mean(dim=0).numpy()


def featurize(sub):
    return np.vstack([target_embedding(s, w) for s, w in zip(sub["sentence"], sub["word"])])


run_cv("bert", df, featurize)
