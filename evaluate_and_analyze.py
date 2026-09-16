"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) System
Module: Comprehensive Evaluation & Error Analysis (Phase 8 & Phase 9)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.stdout.reconfigure(encoding='utf-8')

def run_error_analysis(splits_file: str):
    with open(splits_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train = data['train']
    test = data['test']
    catalog = data['catalog']

    train_by_word = {}
    for r in train:
        train_by_word.setdefault(r['folder'], []).append(r)

    test_by_word = {}
    for r in test:
        test_by_word.setdefault(r['folder'], []).append(r)

    all_errors = []
    word_stats = []

    for folder, test_items in sorted(test_by_word.items()):
        train_items = train_by_word.get(folder, [])
        if not train_items:
            continue

        train_texts = [r['text'] for r in train_items]
        train_labels = [r['sense_label'] for r in train_items]

        test_texts = [r['text'] for r in test_items]
        test_labels = [r['sense_label'] for r in test_items]

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        clf = LinearSVC(C=1.0, loss='squared_hinge', random_state=42, dual='auto')
        clf.fit(X_train, train_labels)

        preds = clf.predict(X_test)
        decision_scores = clf.decision_function(X_test)

        w = catalog[folder]['target_word']
        num_senses = len(catalog[folder]['senses'])

        for idx, (t_item, true_lbl, pred_lbl) in enumerate(zip(test_items, test_labels, preds)):
            if true_lbl != pred_lbl:
                # Calculate approximate confidence from decision function
                if decision_scores.ndim == 1:
                    conf = 1.0 / (1.0 + np.exp(-abs(decision_scores[idx])))
                else:
                    exp_s = np.exp(decision_scores[idx] - np.max(decision_scores[idx]))
                    probs = exp_s / np.sum(exp_s)
                    conf = probs[pred_lbl]

                true_s_num = int(true_lbl) + 1
                pred_s_num = int(pred_lbl) + 1
                true_s_def = catalog[folder]['senses'].get(str(true_s_num), catalog[folder]['senses'].get(true_s_num, ''))
                pred_s_def = catalog[folder]['senses'].get(str(pred_s_num), catalog[folder]['senses'].get(pred_s_num, ''))

                all_errors.append({
                    'folder': folder,
                    'target_word': w,
                    'num_senses': int(num_senses),
                    'context': t_item['text'],
                    'true_sense_num': int(true_s_num),
                    'true_sense_def': str(true_s_def),
                    'predicted_sense_num': int(pred_s_num),
                    'predicted_sense_def': str(pred_s_def),
                    'confidence': float(conf)
                })

        word_acc = accuracy_score(test_labels, preds)
        word_stats.append({
            'folder': folder,
            'target_word': w,
            'num_senses': num_senses,
            'accuracy': word_acc
        })

    print(f"Total test instances evaluated: {len(test)}")
    print(f"Total misclassifications (errors): {len(all_errors)} ({len(all_errors)/len(test)*100:.2f}%)")

    # Save all errors to JSON
    err_path = r"e:\ArthoBodh\baseline_errors.json"
    with open(err_path, 'w', encoding='utf-8') as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)

    # Print representative errors
    print("\n=======================================================")
    print("           PHASE 8: REPRESENTATIVE ERROR ANALYSIS       ")
    print("=======================================================")
    for i, err in enumerate(all_errors[:6], 1):
        print(f"\n--- Error Case {i} ---")
        print(f"Target word:          {err['target_word']} (Word folder: {err['folder']}, {err['num_senses']} senses)")
        print(f"Context:              {err['context'][:150]}...")
        print(f"True sense:           Sense {err['true_sense_num']} — {err['true_sense_def']}")
        print(f"Predicted sense:      Sense {err['predicted_sense_num']} — {err['predicted_sense_def']}")
        print(f"Prediction confidence:{err['confidence']*100:.2f}%")

    return all_errors, word_stats

if __name__ == '__main__':
    splits_file = r"e:\ArthoBodh\dataset_splits.json"
    run_error_analysis(splits_file)
