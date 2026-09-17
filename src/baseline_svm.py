"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) System
Module: TF-IDF + Linear SVM Baseline (comparison for the transformer)

    python -m src.baseline_svm

Implements:
- Word-specific TF-IDF Vectorizer (1-2 word n-grams)
- Per-word Linear Support Vector Machine (LinearSVC)
- Evaluates on the held-out test split (690 instances across 100 words)
- Computes Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Computes breakdown for 3-Sense vs 4-Sense words
- Plots and saves the Confusion Matrix
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def run_baseline_svm(splits_file: str, vis_dir: str):
    with open(splits_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train = data['train']
    val = data['val']
    test = data['test']
    catalog = data['catalog']

    # Group train and test by folder (target word)
    train_by_word = {}
    for r in train:
        train_by_word.setdefault(r['folder'], []).append(r)

    test_by_word = {}
    for r in test:
        test_by_word.setdefault(r['folder'], []).append(r)

    all_y_true = []
    all_y_pred = []
    word_results = {}

    y_true_3s = []
    y_pred_3s = []
    y_true_4s = []
    y_pred_4s = []

    # Train per-word Linear SVM models
    for folder, test_items in sorted(test_by_word.items()):
        train_items = train_by_word.get(folder, [])
        if not train_items:
            continue

        train_texts = [r['text'] for r in train_items]
        train_labels = [r['sense_label'] for r in train_items]

        test_texts = [r['text'] for r in test_items]
        test_labels = [r['sense_label'] for r in test_items]

        num_senses = len(catalog[folder]['senses'])

        # Feature extraction: TF-IDF with Bengali-friendly word/char n-grams
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Train Linear SVM
        clf = LinearSVC(C=1.0, loss='squared_hinge', random_state=42, dual='auto')
        clf.fit(X_train, train_labels)

        preds = clf.predict(X_test)

        all_y_true.extend(test_labels)
        all_y_pred.extend(preds)

        if num_senses == 3:
            y_true_3s.extend(test_labels)
            y_pred_3s.extend(preds)
        elif num_senses == 4:
            y_true_4s.extend(test_labels)
            y_pred_4s.extend(preds)

        word_acc = accuracy_score(test_labels, preds)
        word_results[folder] = {
            'target_word': catalog[folder]['target_word'],
            'num_senses': num_senses,
            'accuracy': word_acc
        }

    # Global Metrics Calculation
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

    print("\n=======================================================")
    print("      ARTHOBODH: TF-IDF + LINEAR SVM BASELINE RESULTS  ")
    print("=======================================================")
    print(f"Total Test Instances: {len(all_y_true)}")
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

    # Confusion Matrix Plot
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    labels = ['Sense 1', 'Sense 2', 'Sense 3', 'Sense 4']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='TF-IDF + Linear SVM: Confusion Matrix (Relative Senses)',
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
    cm_path = os.path.join(vis_dir, 'svm_confusion_matrix.png')
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    # Save metrics JSON
    metrics = {
        'model': 'TF-IDF + Linear SVM',
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
    from src import config
    metrics_dir = config.METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "baseline_svm_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics

if __name__ == '__main__':
    from src import config
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    run_baseline_svm(str(config.SPLITS_PATH), str(config.PLOTS_DIR))
