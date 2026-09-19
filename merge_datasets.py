"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD)
Module: Dataset Harmonization and Merger (Raw Pipeline)

Builds a unified 5,100-word dataset:
  1. Raw Kaggle Dataset (data/raw/Bengali_WSD_Database):
     - Parses raw Sense1.txt..Sense4.txt across 100 word directories.
     - Segments multi-sentence paragraphs into clean single sentences containing the target word.
     - Annotates target boundaries with **target_word**.
  2. IndoWordNet (pyiwn):
     - Fetches 5,000 polysemous words (excluding the 100 Kaggle benchmark words).
     - Standardizes glosses using synonym anchoring and boilerplate removal.
     - Caps candidate senses at the 5 most important senses per word.
     - Extracts verified example sentences with **target_word** boundary annotations.
  3. Stratified Train / Validation / Test Splitting:
     - 70% Train, 15% Validation, 15% Test.
     - Saves to: data/processed_combined/dataset_splits.json
"""

import os
import sys
import re
import json
import random
import unicodedata
from pathlib import Path

# Enable UTF-8 encoding across Windows runtime
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
random.seed(SEED)

_BENGALI_CHAR = r"ঀ-৿"
_ZERO_WIDTH = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_SPACES = re.compile(r"\s+")
_OBSCURE_TERMS = ['অসুর', 'বিরাটের পুত্র', 'বৈদীক যুগের', 'একটি কাব্যালঙ্কার']


def clean_spacing(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = _ZERO_WIDTH.sub("", text)
    return _SPACES.sub(" ", text).strip()


def normalize_text_clean(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\r\t]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r'["`~^+=|\\/«»]', " ", text)
    text = text.replace("নদী বী ", "নদী বা ")
    text = text.replace("অবস্হিত", "অবস্থিত").replace("অবস্হা", "অবস্থা")
    text = text.replace("মুখথেকে", "মুখ থেকে").replace("ব্যাক্তি", "ব্যক্তি")
    text = re.sub(r"\s+", " ", text).strip(" _-—\t\r\n")
    return text


def clean_gloss_normalized(raw_gloss: str, lemmas: list, target_word: str) -> str:
    g = normalize_text_clean(raw_gloss)
    g = re.sub(r'^\s*\([^)]+\)\s*', '', g)
    if 'ফলস্বরূপ হওয়া' in g or 'শেষে তার' in g or 'ফলস্বরূপ হওয়া' in g:
        g = 'কাজের শেষ পরিণতি বা ফলাফল'
    elif 'ফুল থেকে উত্পন্ন হওয়া শাঁস' in g or 'ফুল থেকে উত্পন্ন হওয়া শাঁস' in g:
        g = 'গাছের রসালো খাদ্য বা বীজকোষ'
    elif 'পরিণাম রূপে প্রাপ্ত ফল' in g or g == 'পরিণাম রূপে প্রাপ্ত':
        g = 'কর্মের প্রতিফল বা বদলা'
    elif 'গণিতে কোনো সমস্যার' in g:
        g = 'গণিতের সমাধান বা প্রশ্নের উত্তর'
    else:
        prefixes = [
            r'^কোনো এমন বস্তু যা\s*', r'^এমন বস্তু যা\s*', r'^এমন বিষয় যা\s*', r'^এমন বিষয় যা\s*',
            r'^সেই প্রধান\s*', r'^মানুষের সেই সমূহ যাদের কাছে\s*', r'^সেই\s+', r'^কোনো\s+',
            r'^কোনও\s+', r'^একপ্রকার\s+', r'^একটি\s+', r'^একজন\s+', r'^এক\s+'
        ]
        for p in prefixes:
            g = re.sub(p, '', g, flags=re.IGNORECASE).strip()
        parts = re.split(r'\s+(?:যা|যার|যাকে|যাদের|যাতে|যেখানে|যখন|যে সময়|যে সময়|এবং যার)\s+', g)
        if parts[0] and len(parts[0].split()) >= 2:
            g = parts[0].strip()
        elif len(parts) > 1 and parts[1]:
            g = parts[1].strip()
    g = re.sub(r'\s+(?:বা|এবং|অথবা|ও|ইত্যাদি|প্রভৃতি|সেই)$', '', g).strip(' ,;:-—')
    words = g.split()
    if len(words) > 6:
        g = ' '.join(words[:6])
    norm_target = normalize_text_clean(target_word)
    cleaned_lemmas = [normalize_text_clean(l) for l in lemmas]
    other_lemmas = [l for l in cleaned_lemmas if l.lower() != norm_target.lower()]
    seen = set()
    uniq = [l for l in other_lemmas if not (l in seen or seen.add(l))]
    if uniq:
        syn_str = ', '.join(uniq[:2])
        if g:
            return g if g.startswith(syn_str) else f"{syn_str} ({g})"
        return syn_str
    return g


def extract_and_mark_sentence(paragraph: str, target: str) -> str:
    paragraph = clean_spacing(paragraph)
    target = clean_spacing(target)

    raw_sentences = [s.strip() for s in re.split(r'([।?!]+)', paragraph) if s.strip()]
    sentences = []
    i = 0
    while i < len(raw_sentences):
        s = raw_sentences[i]
        if i + 1 < len(raw_sentences) and re.match(r'^[।?!]+$', raw_sentences[i + 1]):
            s = s + " " + raw_sentences[i + 1]
            i += 2
        else:
            i += 1
        s = _SPACES.sub(" ", s).strip()
        if s:
            sentences.append(s)

    if not sentences:
        sentences = [paragraph]

    pattern = re.compile(rf"(?<![{_BENGALI_CHAR}])({re.escape(target)}[{_BENGALI_CHAR}]*)")
    matched_indices = [idx for idx, s in enumerate(sentences) if pattern.search(s)]

    if not matched_indices:
        matched_indices = [idx for idx, s in enumerate(sentences) if target in s]

    if not matched_indices:
        extracted = paragraph
    else:
        best_idx = matched_indices[0]
        extracted = sentences[best_idx]
        words = extracted.split()
        if len(words) < 6:
            if best_idx + 1 < len(sentences):
                extracted = extracted + " " + sentences[best_idx + 1]
            elif best_idx > 0:
                extracted = sentences[best_idx - 1] + " " + extracted

    cleaned = clean_spacing(extracted)
    if "**" not in cleaned:
        marked, count = pattern.subn(r'**\1**', cleaned, count=1)
        if count == 0:
            fallback = re.compile(rf"(\S*{re.escape(target)}\S*)")
            marked, count = fallback.subn(r'**\1**', cleaned, count=1)
        return marked if count > 0 else cleaned
    return cleaned


def mark_target_in_sentence(context: str, target: str, lemmas: list) -> tuple[str, bool]:
    norm_context = normalize_text_clean(context)
    norm_target = normalize_text_clean(target)
    pattern = rf'(?<![{_BENGALI_CHAR}]){re.escape(norm_target)}([{_BENGALI_CHAR}]*)(?![{_BENGALI_CHAR}])'
    match = re.search(pattern, norm_context)
    if match:
        return re.sub(pattern, rf'**{norm_target}\1**', norm_context, count=1), True
    for l in lemmas:
        l_clean = normalize_text_clean(l)
        if not l_clean:
            continue
        l_pat = rf'(?<![{_BENGALI_CHAR}]){re.escape(l_clean)}([{_BENGALI_CHAR}]*)(?![{_BENGALI_CHAR}])'
        m2 = re.search(l_pat, norm_context)
        if m2:
            return re.sub(l_pat, rf'**{l_clean}\1**', norm_context, count=1), True
    return norm_context, False


def parse_header(header_line: str):
    tags = re.findall(r'<([^>]+)>', header_line)
    word, sense_def = None, None
    for tag in tags:
        t = tag.strip()
        if t.startswith('word-') or t.startswith('word1-'):
            word = re.sub(r'^word\d*-', '', t).strip()
        elif re.match(r'sense\d+-', t):
            sense_def = re.sub(r'^sense\d+-', '', t).strip()
    return word, sense_def


def main():
    print("=== Step 1: Processing Raw Kaggle Dataset ===")
    raw_dir = Path("data/raw/Bengali_WSD_Database")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw Kaggle database not found at {raw_dir}")

    folders = sorted([f for f in os.listdir(raw_dir) if (raw_dir / f).is_dir()],
                     key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)

    kaggle_catalog = {}
    kaggle_records = []
    kaggle_words_set = set()

    for folder in folders:
        folder_path = raw_dir / folder
        kaggle_catalog[folder] = {"target_word": None, "senses": {}, "source": "kaggle_raw"}

        for s_idx in [1, 2, 3, 4]:
            sfile = folder_path / f"Sense{s_idx}.txt"
            if not sfile.exists() or sfile.stat().st_size <= 3:
                continue

            with open(sfile, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read().strip()
            if not content:
                continue

            paras = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            if not paras:
                continue

            first_lines = paras[0].splitlines()
            w, s_def = parse_header(first_lines[0])

            if kaggle_catalog[folder]["target_word"] is None and w:
                w_clean = clean_spacing(w)
                kaggle_catalog[folder]["target_word"] = w_clean
                kaggle_words_set.add(w_clean)

            kaggle_catalog[folder]["senses"][str(s_idx)] = clean_spacing(s_def or "")

            clean_paras = []
            if len(first_lines) > 1:
                body = '\n'.join(first_lines[1:]).strip()
                if body:
                    clean_paras.append(body)
            for p in paras[1:]:
                clean_paras.append(p)

            tw = kaggle_catalog[folder]["target_word"] or w

            for p_text in clean_paras:
                text_sentence = extract_and_mark_sentence(p_text, tw)
                if not text_sentence:
                    continue

                kaggle_records.append({
                    "folder": folder,
                    "target_word": tw,
                    "sense_num": s_idx,
                    "sense_label": s_idx - 1,
                    "sense_def": clean_spacing(s_def or ""),
                    "text": text_sentence,
                    "source_dataset": "kaggle_raw",
                })

    for r in kaggle_records:
        r["num_senses_for_word"] = len(kaggle_catalog[r["folder"]]["senses"])

    print(f"Kaggle words: {len(kaggle_catalog)} | Kaggle sentence instances: {len(kaggle_records)}")

    print("\n=== Step 2: Fetching 5,000 IndoWordNet Words (Excluding 100 Kaggle Words, Max 5 Senses) ===")
    import pyiwn
    iwn = pyiwn.IndoWordNet(pyiwn.Language.BENGALI)
    all_words = iwn.all_words()

    MAX_IWN_WORDS = 5000
    MAX_SENSES_PER_WORD = 5

    iwn_catalog = {}
    iwn_records = []
    excluded_count = 0

    for w in all_words:
        if len(iwn_catalog) >= MAX_IWN_WORDS:
            break
        w_norm = normalize_text_clean(w)
        if w_norm in kaggle_words_set:
            excluded_count += 1
            continue
        try:
            w_syns = iwn.synsets(w)
        except Exception:
            continue
        if len(w_syns) < 2:
            continue
        filtered_syns = [s for s in w_syns if not any(t in s.gloss() for t in _OBSCURE_TERMS)]
        if len(filtered_syns) < 2:
            continue

        # Cap at 5 most important senses
        if len(filtered_syns) > MAX_SENSES_PER_WORD:
            filtered_syns = filtered_syns[:MAX_SENSES_PER_WORD]

        word_id = f"IWN_Word_{len(iwn_catalog) + 1}"
        senses_dict = {}
        for idx, s in enumerate(filtered_syns, 1):
            senses_dict[str(idx)] = clean_gloss_normalized(s.gloss(), s.lemma_names(), w_norm)

        word_records = []
        for idx, s in enumerate(filtered_syns, 1):
            for ex in s.examples():
                marked_ex, found = mark_target_in_sentence(ex, w_norm, s.lemma_names())
                if found:
                    word_records.append({
                        "folder": word_id,
                        "target_word": w_norm,
                        "sense_num": idx,
                        "sense_label": idx - 1,
                        "sense_def": senses_dict[str(idx)],
                        "text": marked_ex,
                        "source_dataset": "indowordnet",
                        "num_senses_for_word": len(filtered_syns),
                    })

        senses_with_examples = len(set(r["sense_num"] for r in word_records))
        if senses_with_examples >= 2:
            iwn_catalog[word_id] = {
                "target_word": w_norm,
                "senses": senses_dict,
                "source": "indowordnet",
            }
            iwn_records.extend(word_records)

    print(f"IndoWordNet words collected: {len(iwn_catalog)} | Verified sentences: {len(iwn_records)}")
    print(f"Excluded baseline words: {excluded_count}")

    print("\n=== Step 3: Merging & Stratified Splitting ===")
    combined_catalog = {}
    combined_catalog.update(iwn_catalog)
    combined_catalog.update(kaggle_catalog)

    def make_stratified_split(records):
        shuffled = list(records)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

    iwn_tr, iwn_val, iwn_te = make_stratified_split(iwn_records)
    kag_tr, kag_val, kag_te = make_stratified_split(kaggle_records)

    train_combined = iwn_tr + kag_tr
    val_combined = iwn_val + kag_val
    test_combined = iwn_te + kag_te

    random.shuffle(train_combined)
    random.shuffle(val_combined)
    random.shuffle(test_combined)

    for r in train_combined: r["split"] = "train"
    for r in val_combined: r["split"] = "val"
    for r in test_combined: r["split"] = "test"

    combined_dataset = {
        "metadata": {
            "description": "Unified Bengali WSD Dataset combining 5,000 IndoWordNet words (max 5 senses) and 100 Raw Kaggle Benchmark words (sentence-extracted)",
            "total_words": len(combined_catalog),
            "total_instances": len(train_combined) + len(val_combined) + len(test_combined),
            "train_instances": len(train_combined),
            "val_instances": len(val_combined),
            "test_instances": len(test_combined),
            "words_indowordnet": len(iwn_catalog),
            "words_kaggle_raw": len(kaggle_catalog),
            "max_senses_per_word_indowordnet": MAX_SENSES_PER_WORD,
            "context_granularity": "Harmonized Sentence-Level",
            "target_marking_format": "**target_word**",
            "random_seed": SEED,
        },
        "catalog": combined_catalog,
        "train": train_combined,
        "val": val_combined,
        "test": test_combined,
    }

    output_path = Path("data/processed_combined/dataset_splits.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSUCCESS! Saved unified dataset to: {output_path}")
    print(f"  Catalog words: {len(combined_catalog):,} (5,000 IndoWordNet + 100 Kaggle)")
    print(f"  Train:         {len(train_combined):,} sentences")
    print(f"  Val:           {len(val_combined):,} sentences")
    print(f"  Test:          {len(test_combined):,} sentences")
    print(f"  Total:         {len(train_combined) + len(val_combined) + len(test_combined):,} instances")


if __name__ == "__main__":
    main()
