"""
ArthoBodh web server.

    GET  /            web interface (web/)
    GET  /metadata    model status, test metrics, supported words
    GET  /examples    a few held-out test sentences to try
    POST /predict     {"sentence", "target_word"} -> ranked senses from the fine-tuned model

The server only serves predictions from the fine-tuned checkpoint in
checkpoints/banglabert-wsd. If it is missing, /predict answers 503 with instructions
instead of returning untrained guesses.
"""

import json
import random
import sys

from flask import Flask, jsonify, request, send_from_directory

from src import config
from src.text import normalize_text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WEB_DIR = config.ROOT / "web"
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")

# ---- sense inventory: target word -> {sense_num: definition} ------------------------
with open(config.SPLITS_PATH, "r", encoding="utf-8") as f:
    _splits = json.load(f)
SENSES = {normalize_text(info["target_word"]): info["senses"] for info in _splits["catalog"].values()}
_EXAMPLES = random.Random(config.SEED).sample(_splits["test"], 12)
del _splits

# ---- fine-tuned model ---------------------------------------------------------------
model, model_error = None, None
if (config.CHECKPOINT_DIR / "config.json").exists():
    try:
        from src.model import GlossWSDModel
        print(f"Loading fine-tuned model from {config.CHECKPOINT_DIR} ...")
        model = GlossWSDModel.load(config.CHECKPOINT_DIR)
        print(f"Model ready on {model.device}.")
    except Exception as e:  # e.g. torch blocked by OS policy, corrupt checkpoint
        model_error = f"Model failed to load: {type(e).__name__}: {e}"
else:
    model_error = ("No fine-tuned model found at checkpoints/banglabert-wsd. "
                   "Train it (python -m src.train, or the Colab notebook) and place the folder there.")
if model_error:
    print(f"[warning] {model_error}")


def _test_metrics():
    path = config.METRICS_DIR / "transformer_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


# ---- routes -------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/metadata")
def metadata():
    return jsonify({
        "model": "BanglaBERT gloss cross-encoder" if model else None,
        "model_ready": model is not None,
        "model_error": model_error,
        "test_metrics": _test_metrics(),
        "supported_words": sorted(SENSES),
    })


@app.route("/examples")
def examples():
    return jsonify([{"sentence": r["text"], "target_word": r["target_word"]} for r in _EXAMPLES])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    sentence = str(data.get("sentence", "")).strip()
    target = normalize_text(data.get("target_word", ""))

    if not sentence or not target:
        return jsonify({"detail": "Please provide both sentence and target_word."}), 400
    if target not in SENSES:
        return jsonify({"detail": f"“{target}” is not one of the {len(SENSES)} words the model was trained on."}), 400
    if model is None:
        return jsonify({"detail": model_error}), 503

    ranked, found = model.predict(sentence, target, SENSES[target])
    if not found:
        return jsonify({"detail": f"The target word “{target}” was not found in the sentence."}), 400

    return jsonify({
        "target_word": target,
        "predicted_sense": ranked[0]["sense"],
        "confidence": ranked[0]["probability"],
        "alternative_senses": [{"sense": r["sense"], "probability": r["probability"]} for r in ranked],
    })
