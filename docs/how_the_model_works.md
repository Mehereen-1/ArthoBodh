# How the ArthoBodh Transformer Works

This document follows **one real sentence** through the trained model, from the moment it is typed into the web page
to the final answer. Every value below (tokens, IDs, tensor shapes, scores, probabilities) was recorded from the
fine-tuned checkpoint in `checkpoints/banglabert-wsd/`.

**Example input:** sentence **"মেয়েটির চোখে জল এসে গেল।"**, target word **জল**

## Overview

```
sentence + word
  → look up definitions → normalize → mark word → build one pair per meaning
  → tokenize → embeddings → 12 transformer layers → [CLS] vector
  → scoring head → one score per meaning → softmax → ranked meanings
```

| Step | Where | Input | Output |
|---|---|---|---|
| 1 | `backend/app.py` | JSON request | definitions of the word |
| 2 | `src/text.py` `normalize_text` | raw text | normalized text |
| 3 | `src/text.py` `mark_target` | sentence, word | sentence with the word in quotes |
| 4 | `src/text.py` `build_pairs` | marked sentence, definitions | 4 text pairs |
| 5 | `src/model.py` `encode` | 4 text pairs | token tensors (4, 17) |
| 6 | BanglaBERT embeddings | token IDs | vectors (4, 17, 768) |
| 7 | 12 transformer layers | (4, 17, 768) | contextual vectors (4, 17, 768) |
| 8 | `[CLS]` position | (4, 17, 768) | (4, 768) |
| 9 | scoring head | (4, 768) | scores (4, 1) |
| 10 | `src/model.py` `score` | 4 scores | matrix (1, 4) |
| 11 | softmax | (1, 4) scores | probabilities |
| 12 | `backend/app.py` → `web/app.js` | probabilities | ranked JSON → page |

---

## Step 1: Request arrives (`backend/app.py`, `predict`)

**Input:** `{"sentence": "মেয়েটির চোখে জল এসে গেল।", "target_word": "জল"}` from the browser.

**What happens:** the server looks up জল in the sense list loaded from `data/processed/dataset_splits.json`.

**Output:**
```
{1: "এক প্রকার পানীয় বিশেষ", 2: "অত্যাধিক পরিশ্রম", 3: "বৃষ্টি", 4: "অশ্রু"}
```

The model never sees the numbers 1–4 as labels. It only sees the **definition texts**.
If the word is not one of the 100 dataset words, the request is rejected at this point.

---

## Step 2: Normalize (`src/text.py`, `normalize_text`)

**Input:** raw sentence, word and definitions.

**What happens:**
- The official BanglaBERT normalizer unifies different Unicode spellings of the same letter.
- Invisible zero-width characters are removed. For example, `জ‍ল` (with a hidden joiner) becomes `জল`,
  which would otherwise never match.

**Output:** `মেয়েটির চোখে জল এসে গেল।` (unchanged here, because this sentence was already clean).

## Step 3: Mark the target word (`mark_target`)

**Input:** the normalized sentence and `জল`.

**What happens:** a regular expression finds each place where the word *starts* a word and wraps it in quotes.
- The word plus any suffix counts: `জলের` → `" জলের "`.
- The same letters in the middle of another word do not count: `সজল` stays unmarked.

**Output:** `মেয়েটির চোখে " জল " এসে গেল।` and `found = True`. If the word is not found, the server rejects the request.

**Why:** a paragraph can contain many words. The quotes tell the model *which* word to disambiguate.

## Step 4: Build one pair per meaning (`build_pairs`)

**Input:** the marked sentence and the 4 definitions.

**Output:** 4 text pairs (A = context, B = candidate):

| | A (context) | B (candidate) |
|---|---|---|
| pair 1 | মেয়েটির চোখে " জল " এসে গেল। | জল : এক প্রকার পানীয় বিশেষ |
| pair 2 | (same) | জল : অত্যাধিক পরিশ্রম |
| pair 3 | (same) | জল : বৃষ্টি |
| pair 4 | (same) | জল : অশ্রু |

The question is no longer "which class is this?" but, four times over, **"does this definition fit this sentence?"**

---

## Step 5: Tokenize (`src/model.py`, `encode`)

**Input:** the 4 text pairs.

**What happens:** BanglaBERT's WordPiece tokenizer (vocabulary of 32,000) splits the text into tokens and converts each token to an ID.

**Output:** three tensors, each of shape **(4, 17)**, meaning 4 pairs × 17 tokens.

Pair 4:
```
tokens:          [CLS] মেয়েটির চোখে  "  জল  "  এসে  গেল  ।  [SEP] জল  :  অশ্রু [SEP] [PAD] [PAD] [PAD]
input_ids:         2   6225    1864  6  1681 6  1054 1092 205   3   1681 30 10280   3     0     0     0
token_type_ids:    0    0       0    0   0   0   0    0    0    0    1   1   1     1     0     0     0
attention_mask:    1    1       1    1   1   1   1    1    1    1    1   1   1     1     0     0     0
```

- **`[CLS]`:** special first token. Its final vector becomes the summary of the whole pair.
- **`[SEP]`:** separates the sentence from the definition.
- **`token_type_ids`:** 0 = sentence part, 1 = definition part.
- **`[PAD]` + `attention_mask`:** pair 1 has a longer definition (17 tokens), so the shorter pairs are padded to the same
  length. Mask 0 tells the model to ignore the padding.
- **জল is ID 1681 in both halves**, so the model can directly link the word in the sentence to the word in front of the definition.
- The maximum length is 288 tokens and only the context is ever truncated. The longest paragraph in the dataset is 249 tokens,
  so in practice nothing is cut.

**Rare or long words are split into pieces** (`##` means "continues the previous piece"):

| Word | Tokens |
|---|---|
| জলের | `জলের` |
| জলাশয়ে | `জলাশয়` `##ে` |
| অশ্রুসিক্ত | `অশ্রু` `##সিক` `##্ত` |
| আটাশে | `আটা` `##শে` |
| বিস্ফোরক | `বিস্ফোরক` |

---

## Step 6: Embeddings (inside BanglaBERT)

**Input:** `input_ids` and `token_type_ids`, each (4, 17).

**What happens:** each token becomes a vector of 768 numbers by adding three learned vectors, followed by normalization:
- **token embedding:** what the token is
- **position embedding:** where it sits (0–511)
- **segment embedding:** sentence part or definition part

**Output:** **(4, 17, 768)**. At this point জল has the *same* vector in every sentence, because it has not seen any context yet.

## Step 7: 12 transformer layers (where context comes in)

**Input:** (4, 17, 768).

**What happens in each of the 12 layers:**
1. **Self-attention (12 heads):** every token looks at every other token in the pair (except padding) and mixes in
   information from the ones that seem relevant. Each head learns a different kind of relationship.
2. **Feed-forward network:** 768 → 3072 → 768, applied to each token.
3. Residual connections and layer normalization keep training stable.

**Output:** still **(4, 17, 768)**, but every vector now depends on its context. The জল vector in "চোখে জল" is different
from the জল vector in "বৃষ্টির জল". Because sentence and definition are in one sequence, the definition tokens also attend to the sentence.

Measured attention, averaged over all layers and heads:

| From definition token | → চোখে | → জল |
|---|:-:|:-:|
| অশ্রু (tears) | **0.024** | **0.033** |
| বৃষ্টি (rain) | 0.012 | 0.019 |

অশ্রু pays about twice as much attention to "চোখে" (eyes) and to জল as বৃষ্টি does, which fits "tears are in the eyes".
Attention is only a rough clue about what the model does, not a full explanation.

The encoder has **110.6 million parameters**, all shared across the 100 words. There are no per-word parameters;
everything the model knows about the 100 words lives in these shared weights.

Architecture of `csebuetnlp/banglabert` (ELECTRA-base discriminator):

| Setting | Value |
|---|---|
| Vocabulary | 32,000 |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Feed-forward size | 3072 |
| Max positions | 512 |

## Step 8: Take the [CLS] vector

**Input:** (4, 17, 768).

**Output:** **(4, 768)**, one summary vector per pair. For pair 4 the first 5 values are `[0.125, 0.068, 0.108, -0.316, -0.022]`.
These numbers are not readable on their own; they encode how well the definition fits the context.

## Step 9: Scoring head

**Input:** (4, 768).

**What happens:** `Linear 768→768` → `GELU` → `Dropout` (active only during training) → `Linear 768→1`.

**Output:** **(4, 1)**, one number per pair:

| পানীয় | পরিশ্রম | বৃষ্টি | অশ্রু |
|:-:|:-:|:-:|:-:|
| -5.218 | -5.292 | -5.014 | **+5.590** |

## Step 10: Group the scores by sentence (`score`)

**Input:** the 4 scores and their positions (`rows = [0, 0, 0, 0]`, `cols = [0, 1, 2, 3]`).

**Output:** a **(1, 4)** matrix `[[-5.218, -5.292, -5.014, 5.590]]`.

With several sentences at once (for example a training batch of 8) this is (8, 4). A word with only 3 meanings leaves
its 4th column at `-inf`, which becomes probability 0.

## Step 11: Softmax

**Input:** `[-5.218, -5.292, -5.014, 5.590]`.

**What happens:** `p_i = e^(score_i) / Σ_j e^(score_j)`. A gap of about 10.6 points becomes near-certainty.

**Output:** `[0.002%, 0.002%, 0.002%, 99.994%]`.

## Step 12: Response (`predict` → `backend/app.py` → `web/app.js`)

**Input:** the probabilities.

**Output:** meanings sorted by probability, sent as JSON:
```json
{"target_word": "জল", "predicted_sense": "অশ্রু", "confidence": 0.99994, "alternative_senses": [ ... ]}
```
The page highlights জল in the sentence and draws a probability bar for each meaning.

---

## During training (`src/train.py`)

Training runs **steps 2–10 exactly as above**. Only the end is different:

1. **Batching:** 8 training paragraphs → about 28 pairs → score matrix (8, 4).
2. **Loss:** cross-entropy = `−log(probability of the correct meaning)`, averaged over the batch.
   - For the example sentence: −log(0.99994) ≈ **0.00006**, almost nothing left to learn.
   - Before training the scoring head is random, all scores are about equal (25% each for 4 meanings), and the loss is −log(0.25) ≈ **1.39**.
3. **Backpropagation:** computes how each of the 110.6M parameters should change to raise the correct pair's score and
   lower the other pairs' scores for the same sentence.
4. **Update:** AdamW (learning rate 2e-5, weight decay 0.01), 10% linear warm-up then linear decay, gradient clipping at 1.0,
   mixed precision on GPU. About 302 batches per epoch.
5. **Model selection:** after each epoch, accuracy is measured on the 345 validation paragraphs. The best epoch is saved
   (epoch 5 for the current model), and training stops after 3 epochs without improvement.

The 690 test paragraphs are never used for training or model selection; `src/evaluate.py` uses them once at the end.

---

## Summary: what happens to the target word

1. It is **normalized** so different spellings match.
2. It is **located** with a regex that allows suffixes.
3. It is **marked with quotes** in the sentence.
4. It is **repeated** in front of every definition (`জল : …`) with the same token ID, so the model can connect the two.
5. It **becomes a context-dependent vector** inside the 12 layers, and every definition token can attend to it and to
   the words around it (চোখে).
6. The **[CLS] summary** of each (sentence, definition) pair becomes **one score**, and the **softmax** across the word's
   definitions picks the meaning.

Because the model compares the sentence with definition *text*, it can also score words it was never trained on,
as long as definitions are provided. For example, given the definitions খাদ্য ফল / ফলাফল, it correctly disambiguated
ফল in four test sentences. That is an informal check, not a measured result.
