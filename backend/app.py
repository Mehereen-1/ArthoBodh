"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) Backend Server
Module: backend.app

Serves:
- Real-time WSD Inference API (/predict) across all 100 polysemous words & 345 senses
- Dataset and model metadata endpoint (/metadata)
- Static web application frontend (web/index.html, styles.css, app.js)

Resilience:
- Automatically loads PyTorch & BanglaBERT if available in the environment.
- Gracefully falls back to a high-speed Pure-Python Contextual Matcher if OS
  security policies (e.g. Windows WDAC / AppLocker) block PyTorch C-extensions.
"""

import os
import sys
import re
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Windows console unicode handling
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Determine project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / 'web'
PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'dataset_splits.json'
FALLBACK_DATA_PATH = PROJECT_ROOT / 'dataset_splits.json'
WEIGHTS_PATH = PROJECT_ROOT / 'best_banglabert_wsd.pt'

MODEL_NAME = 'csebuetnlp/banglabert'

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path='')
CORS(app)

# 1. Load Data Catalog & Training Corpora
split_file = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else FALLBACK_DATA_PATH
if not split_file.exists():
    raise FileNotFoundError(f"Cannot find dataset_splits.json at {PROCESSED_DATA_PATH} or {FALLBACK_DATA_PATH}")

print(f"Loading Bengali WSD Sense Catalog from {split_file}...")
with open(split_file, 'r', encoding='utf-8') as f:
    splits_data = json.load(f)

catalog = splits_data.get('catalog', {})
train_records = splits_data.get('train', [])

# Build fast lookup map: target_word -> {folder, senses}
word_map = {}
for folder, info in catalog.items():
    w = info.get('target_word')
    if w:
        word_map[w.strip()] = {
            'folder': folder,
            'senses': info['senses']
        }

# Also merge prototype senses (e.g. फल) so all examples work seamlessly
TOY_SENSES_PATH = PROJECT_ROOT / 'data' / 'prototype' / 'senses.json'
TOY_DATA_PATH = PROJECT_ROOT / 'data' / 'prototype' / 'data.csv'
if TOY_SENSES_PATH.exists():
    try:
        with open(TOY_SENSES_PATH, 'r', encoding='utf-8') as f:
            toy_senses = json.load(f)
        for tw, ts in toy_senses.items():
            if tw not in word_map:
                formatted_senses = {int(k) + 1: v for k, v in ts.items()}
                word_map[tw] = {
                    'folder': f'prototype_{tw}',
                    'senses': formatted_senses
                }
    except Exception as e:
        print(f"[Notice] Could not load prototype senses: {e}")

# Pre-index training word-context profiles per sense for high-speed fallback
PUNCT_REGEX = re.compile(r"[।॥,;:!?\"'()\[\]\-–—\s]+")
def tokenize(text):
    return [t for t in PUNCT_REGEX.split(text) if t]

sense_profiles = defaultdict(lambda: defaultdict(Counter))
for r in train_records:
    w = r.get('target_word')
    s_num = r.get('sense_num', 1)
    words = tokenize(r.get('text', ''))
    for token in words:
        if token != w:
            sense_profiles[w][s_num][token] += 1

if TOY_DATA_PATH.exists():
    try:
        import pandas as pd
        tdf = pd.read_csv(TOY_DATA_PATH)
        for _, row in tdf.iterrows():
            tw = str(row['word']).strip()
            s_id = int(row['sense_id']) + 1
            for tok in tokenize(str(row['sentence'])):
                if tok != tw:
                    sense_profiles[tw][s_id][tok] += 3
    except Exception:
        pass

# Also index sense definition keywords
for w, winfo in word_map.items():
    for s_num_str, s_def in winfo['senses'].items():
        s_num = int(s_num_str)
        def_tokens = tokenize(s_def)
        for token in def_tokens:
            sense_profiles[w][s_num][token] += 5  # extra weight for definition keywords

print(f"Indexed {len(word_map)} polysemous words and {len(train_records)} training contexts.")

# 2. Try Loading Transformer Model (with graceful fallback)
torch_available = False
tokenizer = None
model = None
device = None
model_status = "Contextual Pure-Python Engine"

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    class BanglaBERTWSDModel(nn.Module):
        def __init__(self, model_name=MODEL_NAME, num_labels=4):
            super().__init__()
            self.transformer = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            self.num_labels = num_labels

        def forward(self, input_ids, attention_mask, token_type_ids=None, num_senses=None):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            if num_senses is not None:
                mask = torch.zeros_like(logits)
                for b, n in enumerate(num_senses):
                    if n == 3:
                        mask[b, 3] = -1e9
                logits = logits + mask
            return logits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Initializing BanglaBERT Transformer...")
    model = BanglaBERTWSDModel(MODEL_NAME, num_labels=4).to(device)

    if WEIGHTS_PATH.exists():
        print(f"Loading fine-tuned checkpoint from {WEIGHTS_PATH}...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model_status = "Fine-Tuned BanglaBERT"
    else:
        print("Fine-tuned checkpoint not found locally; running in zero-shot pretrained mode.")
        model_status = "Pretrained BanglaBERT"

    model.eval()
    torch_available = True
    print("BanglaBERT Model successfully loaded!")

except (ImportError, OSError) as e:
    print(f"\n[Notice] Deep learning backend unavailable locally ({type(e).__name__}: {e}).")
    print("[Notice] Activating Pure-Python Contextual Inference Engine.")
    torch_available = False
    model_status = "Contextual Similarity Engine"


# 3. Prediction Helpers
def predict_with_transformer(sentence, target_word, num_senses):
    enc = tokenizer(
        sentence,
        target_word,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    token_type_ids = enc.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    num_senses_t = torch.tensor([num_senses], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids, num_senses_t)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    valid_probs = [float(p) for p in probs[:num_senses]]
    total = sum(valid_probs)
    return [p / total for p in valid_probs] if total > 0 else [1.0 / num_senses] * num_senses


def predict_with_lexical_engine(sentence, target_word, senses_dict):
    tokens = tokenize(sentence)
    scores = {}
    word_profiles = sense_profiles.get(target_word, {})

    for s_key in senses_dict.keys():
        s_num = int(s_key)
        profile = word_profiles.get(s_num, Counter())
        score = 0.5  # smoothing prior

        for tok in tokens:
            if tok != target_word:
                score += profile.get(tok, 0)
        scores[s_num] = score

    # Softmax conversion
    max_score = max(scores.values()) if scores else 1.0
    exp_scores = {s: math.exp((sc - max_score) / max(1.0, max_score * 0.2)) for s, sc in scores.items()}
    total_exp = sum(exp_scores.values()) or 1.0

    probs_by_sense = {s: exp_scores[s] / total_exp for s in scores}
    return probs_by_sense


# 4. HTTP Routes
@app.route('/')
def serve_index():
    return send_from_directory(str(WEB_DIR), 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    target = WEB_DIR / path
    if target.exists():
        return send_from_directory(str(WEB_DIR), path)
    return send_from_directory(str(WEB_DIR), 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) or {}
    sentence = data.get('sentence', '').strip()
    target_word = data.get('target_word', '').strip()

    if not sentence or not target_word:
        return jsonify({'detail': 'Please provide both sentence and target_word.'}), 400

    # Look up word in catalog
    word_info = word_map.get(target_word)
    if word_info:
        senses_dict = word_info['senses']
        num_senses = len(senses_dict)
    else:
        # Fallback for arbitrary out-of-catalog test words
        num_senses = 4
        senses_dict = {
            1: "অর্থ ১ (সাধারণ প্রয়োগ)",
            2: "অর্থ ২ (আলংকারিক প্রয়োগ)",
            3: "অর্থ ৩ (প্রাসঙ্গিক প্রয়োগ)",
            4: "অর্থ ৪ (বিশেষ্য/বিশেষণ প্রয়োগ)"
        }

    alternatives = []
    if torch_available and model is not None:
        probs = predict_with_transformer(sentence, target_word, num_senses)
        for idx, prob in enumerate(probs):
            s_num = idx + 1
            s_def = senses_dict.get(s_num, senses_dict.get(str(s_num), f"Sense {s_num}"))
            alternatives.append({
                'sense': s_def,
                'probability': float(round(prob, 4))
            })
    else:
        probs_dict = predict_with_lexical_engine(sentence, target_word, senses_dict)
        for s_key in sorted(senses_dict.keys(), key=lambda k: int(k)):
            s_num = int(s_key)
            prob = probs_dict.get(s_num, 1.0 / num_senses)
            s_def = senses_dict.get(s_num, senses_dict.get(str(s_num), f"Sense {s_num}"))
            alternatives.append({
                'sense': s_def,
                'probability': float(round(prob, 4))
            })

    # Sort descending by probability
    alternatives.sort(key=lambda x: -x['probability'])
    top_pred = alternatives[0]

    response = {
        'target_word': target_word,
        'predicted_sense': top_pred['sense'],
        'confidence': top_pred['probability'],
        'alternative_senses': alternatives,
        'model_status': model_status
    }
    return jsonify(response)


@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        'task': 'Bengali Word Sense Disambiguation',
        'dataset': 'Bengali WSD Dataset (100 Polysemous Words, 345 Senses)',
        'model_status': model_status,
        'baseline_accuracy': '69.42% (TF-IDF + Linear SVM)',
        'total_target_words': len(word_map),
        'supported_words': sorted(list(word_map.keys()))
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"\n=======================================================")
    print(f"   ArthoBodh Interface running at: http://127.0.0.1:{port}")
    print(f"   Active Engine: {model_status}")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
