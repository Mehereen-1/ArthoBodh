import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from normalizer import normalize
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import ROOT, context_window, load_data  # noqa: E402

senses = json.loads((ROOT / "data" / "senses.json").read_text(encoding="utf-8"))

# Small corpus -> train one TF-IDF model per word at startup. Swap in the best arm once results are in.
models = {}
for word, sub in load_data().groupby("word"):
    X = [" ".join(context_window(s, word)) for s in sub["sentence"]]
    models[word] = make_pipeline(TfidfVectorizer(token_pattern=r"\S+"),
                                 LogisticRegression(max_iter=1000)).fit(X, sub["sense_id"])

app = FastAPI()


class PredictRequest(BaseModel):
    sentence: str
    target_word: str


@app.post("/predict")
def predict(req: PredictRequest):
    sentence, word = normalize(req.sentence), normalize(req.target_word)
    if word not in models:
        raise HTTPException(404, f"'{word}' is not a supported word. Supported: {', '.join(models)}")
    try:
        context = " ".join(context_window(sentence, word))
    except ValueError as e:
        raise HTTPException(400, str(e))
    pipe = models[word]
    probs = pipe.predict_proba([context])[0]
    alts = sorted(({"sense": senses[word][str(c)], "probability": float(p)}
                   for c, p in zip(pipe.classes_, probs)), key=lambda a: -a["probability"])
    return {"target_word": word, "predicted_sense": alts[0]["sense"],
            "confidence": alts[0]["probability"], "alternative_senses": alts}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
