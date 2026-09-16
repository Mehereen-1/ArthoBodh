"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) Server
Module: Backend API & Static Web Server

Connects the HTML/JS web interface directly to the Transformer (BanglaBERT) model
for instant real-time inference without retraining.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__, static_folder='.')
CORS(app)

MODEL_NAME = 'csebuetnlp/banglabert'
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_banglabert_wsd.pt')
SPLITS_PATH = os.path.join(os.path.dirname(__file__), 'dataset_splits.json')

# 1. Define Model Architecture
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

# 2. Global State Initialization
print("Loading Bengali WSD Sense Catalog...")
with open(SPLITS_PATH, 'r', encoding='utf-8') as f:
    splits_data = json.load(f)
catalog = splits_data['catalog']

# Build fast lookup map: target_word -> {folder, senses}
word_map = {}
for folder, info in catalog.items():
    w = info['target_word']
    if w:
        word_map[w.strip()] = {
            'folder': folder,
            'senses': info['senses']
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference device: {device}")

print(f"Loading Tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Initializing BanglaBERT Transformer...")
model = BanglaBERTWSDModel(MODEL_NAME, num_labels=4).to(device)

if os.path.exists(WEIGHTS_PATH):
    print(f"Loading fine-tuned checkpoint from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model_status = "Fine-Tuned BanglaBERT"
else:
    print("Fine-tuned checkpoint not found locally; running in zero-shot pretrained mode.")
    model_status = "Pretrained BanglaBERT"

model.eval()
print("Transformer Model ready for inference!")

# 3. Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentence = data.get('sentence', '').strip()
    target_word = data.get('target_word', '').strip()

    if not sentence or not target_word:
        return jsonify({'error': 'Please provide both sentence and target_word.'}), 400

    # Look up word in catalog
    word_info = word_map.get(target_word)
    if word_info:
        senses_dict = word_info['senses']
        num_senses = len(senses_dict)
    else:
        # Fallback if user tests an out-of-vocabulary word
        num_senses = 4
        senses_dict = {
            1: "অর্থ ১ (সাধারণ ভাব)",
            2: "অর্থ ২ (আলংকারিক ভাব)",
            3: "অর্থ ৩ (প্রাসঙ্গিক ভাব)",
            4: "অর্থ ৪ (বিশেষ্য/বিশেষণ ভাব)"
        }

    # Cross-encoder tokenization: context [SEP] target_word
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
    token_type_ids = enc['token_type_ids'].to(device) if 'token_type_ids' in enc else None
    num_senses_t = torch.tensor([num_senses], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids, num_senses_t)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Slice probabilities to valid sense count
    valid_probs = probs[:num_senses]
    valid_probs = valid_probs / np.sum(valid_probs)  # normalize

    ranked_indices = np.argsort(valid_probs)[::-1]
    top_idx = ranked_indices[0]
    top_sense_num = int(top_idx + 1)
    predicted_sense_def = senses_dict.get(top_sense_num, senses_dict.get(str(top_sense_num), f"Sense {top_sense_num}"))

    # Format alternatives list matching app.js expectations
    alternatives = []
    for idx in ranked_indices:
        s_num = int(idx + 1)
        s_def = senses_dict.get(s_num, senses_dict.get(str(s_num), f"Sense {s_num}"))
        alternatives.append({
            'sense': s_def,
            'probability': float(round(valid_probs[idx], 4))
        })

    response = {
        'target_word': target_word,
        'predicted_sense': predicted_sense_def,
        'confidence': float(round(valid_probs[top_idx], 4)),
        'alternative_senses': alternatives,
        'model_status': model_status
    }
    return jsonify(response)

@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        'task': 'Bengali Word Sense Disambiguation',
        'dataset': 'Bengali WSD Dataset (100 Polysemous Words, 345 Senses)',
        'model': 'csebuetnlp/banglabert (Electra Discriminator)',
        'baseline_accuracy': '69.42% (TF-IDF + Linear SVM)',
        'total_target_words': len(word_map),
        'supported_words': sorted(list(word_map.keys()))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"\n=======================================================")
    print(f"   ArthoBodh Interface running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
