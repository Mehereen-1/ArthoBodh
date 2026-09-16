"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) System
Module: Data Loader and Split Generator

Handles:
- Dataset discovery and loading across 100 polysemous word folders
- Parsing XML-like header tags: <word-...><senseN-...>
- Segmenting context paragraphs
- Computing length statistics
- Creating reproducible, stratified Train / Validation / Test splits
"""

import os
import re
import json
import random
from collections import Counter
import pandas as pd
import numpy as np


def parse_header(header_line: str):
    """
    Parses headers such as:
      <word-জল >< sense1-এক প্রকার পানীয় বিশেষ >
      <word1-ছানি ><sense1-ক্ষতিকারক চোখের রোগ>
    Returns (target_word, sense_definition).
    """
    tags = re.findall(r'<([^>]+)>', header_line)
    word, sense_def = None, None
    for tag in tags:
        t = tag.strip()
        if t.startswith('word-') or t.startswith('word1-'):
            word = re.sub(r'^word\d*-', '', t).strip()
        elif re.match(r'sense\d+-', t):
            sense_def = re.sub(r'^sense\d+-', '', t).strip()
    return word, sense_def


def clean_bengali_text(text: str) -> str:
    """
    Cleans and normalizes Bengali text while preserving:
    - Bengali unicode characters and diacritics
    - Punctuation (dari, question marks, commas, quotes)
    - Replaces internal line breaks and multiple spaces with a single space.
    """
    text = text.replace('\ufeff', '').replace('\u200b', '')
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ' '.join(lines).strip()


def load_bengali_wsd_dataset(data_dir: str):
    """
    Loads all records from the 100 word directories.
    Returns:
      records: List[dict]
      word_sense_catalog: Dict[str, dict]
    """
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    records = []
    word_sense_catalog = {}

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        word_sense_catalog[folder] = {
            'target_word': None,
            'senses': {}
        }

        for s_idx in [1, 2, 3, 4]:
            sfile = f"Sense{s_idx}.txt"
            fp = os.path.join(folder_path, sfile)
            if not os.path.exists(fp) or os.path.getsize(fp) <= 3:
                continue

            with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
                content = f.read().strip()
            if not content:
                continue

            paras = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            if not paras:
                continue

            first_lines = paras[0].splitlines()
            w, s_def = parse_header(first_lines[0])

            if word_sense_catalog[folder]['target_word'] is None and w:
                word_sense_catalog[folder]['target_word'] = w

            word_sense_catalog[folder]['senses'][s_idx] = s_def

            clean_paras = []
            if len(first_lines) > 1:
                body = '\n'.join(first_lines[1:]).strip()
                if body:
                    clean_paras.append(body)
            for p in paras[1:]:
                clean_paras.append(p)

            for p_i, p_text in enumerate(clean_paras):
                cleaned_text = clean_bengali_text(p_text)
                if not cleaned_text:
                    continue

                rec = {
                    'folder': folder,
                    'target_word': w,
                    'sense_num': s_idx,           # 1-indexed (1, 2, 3, 4)
                    'sense_label': s_idx - 1,     # 0-indexed (0, 1, 2, 3)
                    'sense_def': s_def,
                    'sense_id': f"{folder}_S{s_idx}",
                    'para_index': p_i,
                    'text': cleaned_text,
                    'char_len': len(cleaned_text),
                    'word_len': len(cleaned_text.split()),
                    'has_exact_target': (w in cleaned_text if w else False)
                }
                records.append(rec)

    for r in records:
        folder = r['folder']
        r['num_senses_for_word'] = len(word_sense_catalog[folder]['senses'])

    return records, word_sense_catalog


def create_stratified_splits(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42):
    """
    Creates deterministic, stratified splits by (folder, sense_num).
    With 10 examples per sense:
      - 7 examples -> Train
      - 1 example  -> Validation
      - 2 examples -> Test
    """
    random.seed(seed)
    np.random.seed(seed)

    groups = {}
    for r in records:
        key = (r['folder'], r['sense_num'])
        groups.setdefault(key, []).append(r)

    train_set = []
    val_set = []
    test_set = []

    for key, items in sorted(groups.items()):
        shuffled = list(items)
        random.shuffle(shuffled)
        n = len(shuffled)

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val
        if n_train <= 0:
            n_train = 1
            n_val = 0

        train_items = shuffled[:n_train]
        val_items = shuffled[n_train:n_train + n_val]
        test_items = shuffled[n_train + n_val:]

        for item in train_items:
            item_copy = dict(item)
            item_copy['split'] = 'train'
            train_set.append(item_copy)

        for item in val_items:
            item_copy = dict(item)
            item_copy['split'] = 'val'
            val_set.append(item_copy)

        for item in test_items:
            item_copy = dict(item)
            item_copy['split'] = 'test'
            test_set.append(item_copy)

    return train_set, val_set, test_set


if __name__ == '__main__':
    data_dir = r"e:\ArthoBodh\Database - Bengali Word Sense Disambiguation"
    print("Loading Bengali WSD Dataset...")
    records, catalog = load_bengali_wsd_dataset(data_dir)
    print(f"Loaded {len(records)} total records across {len(catalog)} words.")

    words_3s = sum(1 for w, d in catalog.items() if len(d['senses']) == 3)
    words_4s = sum(1 for w, d in catalog.items() if len(d['senses']) == 4)
    print(f"Words with 3 senses: {words_3s}")
    print(f"Words with 4 senses: {words_4s}")
    print(f"Total sense definitions: {3 * words_3s + 4 * words_4s}")

    train, val, test = create_stratified_splits(records, seed=42)
    print(f"Train size: {len(train)} ({len(train)/len(records)*100:.1f}%)")
    print(f"Val size:   {len(val)} ({len(val)/len(records)*100:.1f}%)")
    print(f"Test size:  {len(test)} ({len(test)/len(records)*100:.1f}%)")

    splits_file = r"e:\ArthoBodh\dataset_splits.json"
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump({'train': train, 'val': val, 'test': test, 'catalog': catalog}, f, ensure_ascii=False, indent=2)
    print(f"Saved synchronized dataset splits to {splits_file}")
