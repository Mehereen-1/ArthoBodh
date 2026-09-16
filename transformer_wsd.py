"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) System
Module: Pretrained Bengali Transformer (BanglaBERT) WSD Pipeline

Covers:
- Phase 3: Text Preprocessing & Tokenization Inspection
- Phase 5: Transformer Model Architecture (BanglaBERT with Candidate Sense Masking)
- Phase 6: Training & Validation Loop with Early Stopping
- Phase 7: Held-out Test Set Evaluation & 3 vs 4 Sense Complexity Analysis
- Phase 8: Error Analysis Export
"""

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# Set deterministic seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

MODEL_NAME = 'csebuetnlp/banglabert'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

class BengaliWSDDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        context = item['text']
        target = item['target_word']
        label = item['sense_label']  # 0, 1, 2, or 3
        num_senses = item['num_senses_for_word']  # 3 or 4

        # Cross-encoder input: context [SEP] target_word
        enc = self.tokenizer(
            context,
            target,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0) if 'token_type_ids' in enc else torch.zeros(self.max_len, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'num_senses': torch.tensor(num_senses, dtype=torch.long),
            'folder': item['folder'],
            'target_word': target,
            'sense_num': item['sense_num'],
            'sense_def': item['sense_def'],
            'text': context
        }


def print_preprocessing_examples(records, tokenizer, num_examples=3):
    """
    Phase 3 Requirement:
    Print several processed examples showing:
    - Original sentence / context
    - Target word
    - True sense definition
    - Tokenized representation
    - Target token position(s)
    """
    print("\n=======================================================")
    print("      PHASE 3: PREPROCESSING & TOKENIZATION INSPECTION  ")
    print("=======================================================")
    samples = records[:num_examples]
    for i, item in enumerate(samples, 1):
        context = item['text']
        target = item['target_word']
        sense_def = item['sense_def']
        sense_num = item['sense_num']

        encoded = tokenizer(context, target, truncation=True, max_length=128)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])

        # Find target token positions
        target_token_ids = tokenizer.encode(target, add_special_tokens=False)
        target_tokens = tokenizer.convert_ids_to_tokens(target_token_ids)

        positions = []
        for idx in range(len(tokens) - len(target_tokens) + 1):
            if tokens[idx:idx + len(target_tokens)] == target_tokens:
                positions.append(list(range(idx, idx + len(target_tokens))))

        print(f"\n--- Example {i} ---")
        print(f"Target Word:      {target}")
        print(f"True Sense:       Sense {sense_num} ({sense_def})")
        print(f"Context Snippet:  {context[:120]}...")
        print(f"Token Count:      {len(tokens)}")
        print(f"First 15 Tokens:  {tokens[:15]}")
        print(f"Target Sub-tokens: {target_tokens}")
        print(f"Target Position(s) in Sequence: {positions}")
    print("=======================================================\n")


class BanglaBERTWSDModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=4, dropout=0.2):
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
        logits = outputs.logits  # [batch_size, 4]

        # Mask invalid senses for words that have only 3 senses
        if num_senses is not None:
            mask = torch.zeros_like(logits)
            for b, n in enumerate(num_senses):
                if n == 3:
                    # Invalidate sense 4 (index 3)
                    mask[b, 3] = -1e9
            logits = logits + mask

        return logits


def train_model(train_loader, val_loader, model, device, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print("\n=======================================================")
    print("      PHASE 6: BANGLABERT WSD MODEL TRAINING           ")
    print("=======================================================")
    print(f"Pretrained Model:       {MODEL_NAME}")
    print(f"Optimizer:              AdamW (lr={lr}, weight_decay={WEIGHT_DECAY})")
    print(f"Batch Size:             {BATCH_SIZE}")
    print(f"Max Sequence Length:    {MAX_LEN}")
    print(f"Total Epochs:           {epochs}")
    print(f"Random Seed:            {SEED}")
    print(f"Device:                 {device}")
    print("-------------------------------------------------------")

    best_val_acc = 0.0
    best_weights_path = r"e:\ArthoBodh\best_banglabert_wsd.pt"
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        train_preds, train_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            num_senses = batch['num_senses'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids, num_senses)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_labels)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                num_senses = batch['num_senses'].to(device)

                logits = model(input_ids, attention_mask, token_type_ids, num_senses)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved new best model with Val Acc: {val_acc*100:.2f}%")

    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model weights saved to {best_weights_path}")
    return history, best_weights_path


def evaluate_test(test_loader, model, device, vis_dir):
    print("\n=======================================================")
    print("      PHASE 7: FINAL TEST EVALUATION (Held-Out Test Set)")
    print("=======================================================")
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_confidences = []
    test_records_output = []

    y_true_3s, y_pred_3s = [], []
    y_true_4s, y_pred_4s = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            num_senses = batch['num_senses'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids, num_senses)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confs, _ = torch.max(probs, dim=-1)

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            confs_np = confs.cpu().numpy()
            num_senses_np = num_senses.cpu().numpy()

            all_y_true.extend(labels_np)
            all_y_pred.extend(preds_np)
            all_confidences.extend(confs_np)

            for i in range(len(labels_np)):
                ns = int(num_senses_np[i])
                rec_info = {
                    'folder': batch['folder'][i],
                    'target_word': batch['target_word'][i],
                    'context': batch['text'][i],
                    'true_sense_label': int(labels_np[i]),
                    'true_sense_num': int(batch['sense_num'][i]),
                    'true_sense_def': batch['sense_def'][i],
                    'predicted_sense_label': int(preds_np[i]),
                    'predicted_sense_num': int(preds_np[i] + 1),
                    'confidence': float(confs_np[i]),
                    'is_correct': bool(labels_np[i] == preds_np[i]),
                    'num_senses': ns
                }
                test_records_output.append(rec_info)

                if ns == 3:
                    y_true_3s.append(labels_np[i])
                    y_pred_3s.append(preds_np[i])
                elif ns == 4:
                    y_true_4s.append(labels_np[i])
                    y_pred_4s.append(preds_np[i])

    # Overall metrics
    acc = accuracy_score(all_y_true, all_y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro')
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='weighted')

    # Complexity Breakdown
    acc_3s = accuracy_score(y_true_3s, y_pred_3s)
    _, _, f1_3s_macro, _ = precision_recall_fscore_support(y_true_3s, y_pred_3s, average='macro')
    _, _, f1_3s_wt, _ = precision_recall_fscore_support(y_true_3s, y_pred_3s, average='weighted')

    acc_4s = accuracy_score(y_true_4s, y_pred_4s)
    _, _, f1_4s_macro, _ = precision_recall_fscore_support(y_true_4s, y_pred_4s, average='macro')
    _, _, f1_4s_wt, _ = precision_recall_fscore_support(y_true_4s, y_pred_4s, average='weighted')

    print(f"Total Test Instances:     {len(all_y_true)}")
    print(f"Overall Test Accuracy:    {acc * 100:.2f}%")
    print(f"Macro Precision:          {prec_macro * 100:.2f}%")
    print(f"Macro Recall:             {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:           {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:        {f1_wt * 100:.2f}%")
    print("-------------------------------------------------------")
    print("Word Complexity Breakdown:")
    print(f"  - 3-Sense Words (55 words, N={len(y_true_3s)}): Accuracy = {acc_3s*100:.2f}%, Macro F1 = {f1_3s_macro*100:.2f}%, Weighted F1 = {f1_3s_wt*100:.2f}%")
    print(f"  - 4-Sense Words (45 words, N={len(y_true_4s)}): Accuracy = {acc_4s*100:.2f}%, Macro F1 = {f1_4s_macro*100:.2f}%, Weighted F1 = {f1_4s_wt*100:.2f}%")
    print("=======================================================\n")

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    labels = ['Sense 1', 'Sense 2', 'Sense 3', 'Sense 4']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='BanglaBERT WSD: Confusion Matrix (Relative Senses)',
           ylabel='True Sense',
           xlabel='Predicted Sense')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(vis_dir, 'transformer_confusion_matrix.png')
    fig.savefig(cm_path)
    plt.close(fig)

    metrics = {
        'model': 'BanglaBERT (csebuetnlp/banglabert)',
        'test_instances': len(all_y_true),
        'accuracy': float(acc),
        'macro_precision': float(prec_macro),
        'macro_recall': float(rec_macro),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_wt),
        'three_senses': {
            'instances': len(y_true_3s),
            'accuracy': float(acc_3s),
            'macro_f1': float(f1_3s_macro),
            'weighted_f1': float(f1_3s_wt)
        },
        'four_senses': {
            'instances': len(y_true_4s),
            'accuracy': float(acc_4s),
            'macro_f1': float(f1_4s_macro),
            'weighted_f1': float(f1_4s_wt)
        },
        'confusion_matrix': cm.tolist()
    }

    # Save outputs
    with open(r"e:\ArthoBodh\transformer_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(r"e:\ArthoBodh\transformer_test_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(test_records_output, f, ensure_ascii=False, indent=2)

    return metrics, test_records_output


def run_pipeline():
    splits_file = r"e:\ArthoBodh\dataset_splits.json"
    vis_dir = r"e:\ArthoBodh\visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    with open(splits_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_records = data['train']
    val_records = data['val']
    test_records = data['test']

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Phase 3 Inspection
    print_preprocessing_examples(train_records, tokenizer, num_examples=3)

    # Prepare datasets & loaders
    train_ds = BengaliWSDDataset(train_records, tokenizer, max_len=MAX_LEN)
    val_ds = BengaliWSDDataset(val_records, tokenizer, max_len=MAX_LEN)
    test_ds = BengaliWSDDataset(test_records, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BanglaBERTWSDModel(MODEL_NAME, num_labels=4).to(device)

    # Train
    history, best_weights_path = train_model(train_loader, val_loader, model, device, epochs=EPOCHS, lr=LR)

    # Plot training/validation curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    epochs_range = [h['epoch'] for h in history]
    ax1.plot(epochs_range, [h['train_loss'] for h in history], 'o-', label='Train Loss', color='#e41a1c')
    ax1.plot(epochs_range, [h['val_loss'] for h in history], 's-', label='Val Loss', color='#377eb8')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()

    ax2.plot(epochs_range, [h['train_acc']*100 for h in history], 'o-', label='Train Acc', color='#4daf4a')
    ax2.plot(epochs_range, [h['val_acc']*100 for h in history], 's-', label='Val Acc', color='#984ea3')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    curve_path = os.path.join(vis_dir, 'training_validation_curves.png')
    fig.savefig(curve_path)
    plt.close(fig)

    # Load best model for test evaluation
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    metrics, predictions = evaluate_test(test_loader, model, device, vis_dir)

    print("Pipeline execution completed successfully.")

if __name__ == '__main__':
    run_pipeline()
