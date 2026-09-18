"""
ArthoBodh web server.

    GET  /            web interface (web/)
    GET  /metadata    model status, test metrics, supported words
    GET  /examples    a few held-out test sentences to try
    GET  /dictionary  ?word=...[&pos=noun|verb|adjective|adverb] -> candidate senses
    POST /predict     {"sentence", "target_word", optional "pos"} -> ranked senses from the fine-tuned model

The server only serves predictions from the fine-tuned checkpoint in
checkpoints/banglabert-wsd. If it is missing, /predict answers 503 with instructions
instead of returning untrained guesses.
"""

import json
import random
import sys
import time

from flask import Flask, jsonify, request, send_from_directory

from src import config
from src.dictionary import BengaliDictionary
from src.wordnet_senses import POS_TAGS
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

# Multi-tier dictionary service (Catalog -> IndoWordNet -> Custom)
dict_service = BengaliDictionary(catalog_senses=SENSES)

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
        "external_dictionary": "IndoWordNet (Bengali)",
        "indowordnet_active": dict_service.wordnet_available(),
    })


@app.route("/dictionary", methods=["GET"])
def get_dictionary_senses():
    word = request.args.get("word", "").strip()
    pos = request.args.get("pos") or None
    if not word:
        return jsonify({"detail": "Please provide a 'word' query parameter."}), 400
    if pos is not None and pos not in POS_TAGS:
        return jsonify({"detail": f"pos must be one of: {', '.join(POS_TAGS)}."}), 400
    res = dict_service.get_senses(word, pos=pos)
    return jsonify({
        "word": word,
        "senses": res["senses"],
        "source": res["source"],
        "is_monosemous": res["is_monosemous"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.time()
    data = request.get_json(silent=True) or {}
    sentence = str(data.get("sentence", "")).strip()
    target = normalize_text(data.get("target_word", ""))
    pos = data.get("pos") or None

    if not sentence or not target:
        return jsonify({"detail": "Please provide both sentence and target_word."}), 400
    if pos is not None and pos not in POS_TAGS:
        return jsonify({"detail": f"pos must be one of: {', '.join(POS_TAGS)}."}), 400

    print(f"\n{'='*65}")
    print(f"[REQUEST] Target Word: '{target}'")
    print(f"          Sentence:    '{sentence[:60]}...'")

    # Resolve candidate senses: Catalog -> IndoWordNet
    sense_info = dict_service.get_senses(target, pos=pos)
    senses = sense_info["senses"]
    source = sense_info["source"]
    is_monosemous = sense_info["is_monosemous"]

    if source == "catalog":
        dict_label = "Curated 100-Word Catalog"
        print(f"  ├── [DICTIONARY] Curated Catalog used ({len(senses)} senses)")
    elif source == "supplementary":
        dict_label = "Concise Everyday Lexicon"
        print(f"  ├── [DICTIONARY] Concise Everyday Lexicon used ({len(senses)} senses)")
    elif source == "indowordnet":
        dict_label = "External Library: IndoWordNet (pyiwn)"
        print(f"  ├── [DICTIONARY] *** EXTERNAL LIBRARY USED *** -> Cleaned IndoWordNet ({len(senses)} senses retrieved)")
    else:
        print(f"  └── [DICTIONARY] Word '{target}' NOT found in catalog, supplementary, or IndoWordNet!")
        return jsonify({
            "detail": f"“{target}” was not found in the trained catalog, supplementary lexicon, or IndoWordNet dictionary.",
            "source": "not_found",
        }), 404

    # Check if target appears in the sentence before calling the model
    marked_sample, found = from_target_exists(sentence, target)
    if not found:
        print(f"  └── [VALIDATION] Target '{target}' not found in sentence!")
        return jsonify({"detail": f"The target word “{target}” was not found in the sentence."}), 400

    # CASE A: Monosemous word (only 1 meaning) -> Fast path
    if is_monosemous:
        only_sense = list(senses.values())[0]
        duration_ms = round((time.time() - t_start) * 1000, 1)
        print(f"  ├── [MODEL STATUS] NEURAL MODEL BYPASSED (Word is unambiguous, 1 sense)")
        print(f"  └── [RESULT] '{only_sense}' (Confidence: 100% | Time: {duration_ms}ms)")
        print(f"{'='*65}")
        return jsonify({
            "target_word": target,
            "predicted_sense": only_sense,
            "confidence": 1.0,
            "source": source,
            "is_monosemous": True,
            "note": "Word is unambiguous in dictionary (only one definition exists).",
            "alternative_senses": [{"sense": only_sense, "probability": 1.0}],
            "execution_trace": {
                "dictionary_used": dict_label,
                "external_library_used": source == "indowordnet",
                "neural_model_used": False,
                "neural_model_status": "Bypassed (Only 1 meaning in dictionary)",
                "candidate_senses_count": 1,
                "duration_ms": duration_ms,
            },
        })

    # CASE B: Ambiguous word (2+ meanings) -> Run Cross-Encoder
    if model is None:
        print(f"  └── [MODEL STATUS] Error: No checkpoint found at checkpoints/banglabert-wsd!")
        return jsonify({"detail": model_error}), 503

    print(f"  ├── [MODEL STATUS] *** BANGLABERT NEURAL MODEL USED *** (Cross-attention across {len(senses)} candidate pairs)")
    ranked, found = model.predict(sentence, target, senses)
    if not found:
        return jsonify({"detail": f"The target word “{target}” was not found in the sentence."}), 400

    duration_ms = round((time.time() - t_start) * 1000, 1)
    print(f"  └── [RESULT] Predicted: '{ranked[0]['sense']}' (Confidence: {ranked[0]['probability']*100:.1f}% | Time: {duration_ms}ms)")
    print(f"{'='*65}")

    return jsonify({
        "target_word": target,
        "predicted_sense": ranked[0]["sense"],
        "confidence": ranked[0]["probability"],
        "source": source,
        "is_monosemous": False,
        "alternative_senses": [{"sense": r["sense"], "probability": r["probability"]} for r in ranked],
        "execution_trace": {
            "dictionary_used": dict_label,
            "external_library_used": source == "indowordnet",
            "neural_model_used": True,
            "neural_model_status": f"BanglaBERT Cross-Encoder ({len(senses)} pairs scored)",
            "candidate_senses_count": len(senses),
            "duration_ms": duration_ms,
        },
    })


def from_target_exists(sentence: str, target: str) -> tuple[str, bool]:
    from src.text import mark_target
    return mark_target(sentence, target)
