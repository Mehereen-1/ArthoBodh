# ArthoBodh (অর্থবোধ)
### Bengali Word Sense Disambiguation with a fine-tuned BanglaBERT

Many Bengali words have several unrelated meanings. **জল** can mean *drinking water*, *rain*, *tears* or *hard toil*.
ArthoBodh reads a sentence, looks at the ambiguous word, and tells you which meaning is being used.

```
"মেয়েটির চোখে জল এসে গেল।"  +  জল   →   অশ্রু (tears)
```

---

## 1. The data

`data/raw/Bengali_WSD_Database/` has **100 ambiguous words** (55 with 3 meanings, 45 with 4, so **345 meanings** in total).
Each meaning is a `SenseN.txt` file:

```
<word-জল >< sense3-বৃষ্টি >          ← header: the word and a short definition of this meaning
paragraph using জল to mean rain ...
                                    ← blank line
another paragraph ...               (about 10 paragraphs per meaning)
```

`python -m src.data_loader` parses these files into **3451 labelled paragraphs** and splits every meaning
7 / 1 / 2 into **train (2416) / validation (345) / test (690)**. The result is saved to `data/processed/dataset_splits.json`
together with a `catalog` of every word's definitions. The split is fixed (seed 42), so every model is compared on exactly the same test paragraphs.

---

## 2. How the model works

### The idea: match the context against each definition

Instead of a classifier that outputs "meaning #2" (a label that means something different for every word),
the model is shown **the context together with one candidate definition** and asks: *does this definition fit here?*
It does this for every definition of the word and picks the best fit. This setup is known as a *gloss cross-encoder*
(GlossBERT, Huang et al. 2019). The definitions themselves carry meaning (বৃষ্টি, অশ্রু, ...),
so the model can use what BanglaBERT already knows about those words. That matters here, with only about 7 training paragraphs per meaning.

### Step by step, for one input

**Step 1: Normalize** (`src/text.py → normalize_text`)
Applies the official BanglaBERT normalizer and removes invisible zero-width characters, so `জ‍ল` (with a hidden joiner) and `জল` are the same word.

**Step 2: Mark the target word** (`mark_target`)
Wraps each occurrence of the word in quotes, so the model knows *which* word to disambiguate:
`মেয়েটির চোখে " জল " এসে গেল।`
Inflected forms are included (`জলের` → `" জলের "`), but the word inside another word is not (`সজল` stays unmarked).

**Step 3: Build one pair per meaning** (`build_pairs`)

| | Segment A (context) | Segment B (candidate) |
|---|---|---|
| pair 1 | মেয়েটির চোখে " জল " এসে গেল। | জল : এক প্রকার পানীয় বিশেষ |
| pair 2 | মেয়েটির চোখে " জল " এসে গেল। | জল : অত্যাধিক পরিশ্রম |
| pair 3 | মেয়েটির চোখে " জল " এসে গেল। | জল : বৃষ্টি |
| pair 4 | মেয়েটির চোখে " জল " এসে গেল। | জল : অশ্রু |

**Step 4: Tokenize** (`src/model.py → encode`)
Each pair becomes `[CLS] context tokens [SEP] word : definition tokens [SEP]` (max 288 tokens; only the context
is ever truncated, and the longest paragraph in the dataset is 249 tokens, so in practice nothing is cut).

**Step 5: BanglaBERT reads each pair** (`score`)
`csebuetnlp/banglabert` (an ELECTRA-base encoder: 12 transformer layers, 768-dim, pretrained on large Bengali text)
processes the whole pair at once. Self-attention lets every context token look at every definition token, so the
model can notice, for example, that চোখে ("in the eyes") goes with অশ্রু ("tears").
The `[CLS]` output vector goes through a small head (dense → GELU → linear) that outputs **one number: the match score**.

**Step 6: Softmax over that word's meanings**
The 3 or 4 scores are placed side by side and turned into probabilities:

```
scores (illustrative) [-2.1,  -3.0,  -0.4,   2.7]
softmax  →            [0.8%,  0.3%,  4.5%,  94.4%]   → prediction: অশ্রু
```

A word with only 3 meanings simply has 3 scores; nothing needs to be masked.

### How it learns (`src/train.py`)

1. Take a batch of 8 training paragraphs; expand each into its 3 or 4 pairs (about 28 pairs).
2. Run steps 1–6 to get a probability for every meaning.
3. **Loss = cross-entropy** against the true meaning, i.e. `−log P(correct definition)`. Minimizing it
   raises the correct definition's score and lowers the others *for that same context*.
4. Backpropagate through the head **and all of BanglaBERT** (full fine-tuning), with AdamW (lr 2e-5, weight decay 0.01),
   10% linear warm-up then linear decay, gradient clipping at 1.0, and fp16 on GPU.
5. After each epoch, measure accuracy on the 345 validation paragraphs. The best epoch is saved to
   `checkpoints/banglabert-wsd/`; training stops after 3 epochs without improvement (max 10).

The test set is never used during training; `src/evaluate.py` touches it once at the end.

---

## 3. Project layout

```
ArthoBodh/
├── data/
│   ├── raw/Bengali_WSD_Database/   100 words × SenseN.txt
│   └── processed/dataset_splits.json train / val / test + sense catalog
├── src/
│   ├── config.py          paths and hyperparameters
│   ├── data_loader.py     raw files → dataset_splits.json
│   ├── text.py            normalization, target marking, (context, definition) pairs
│   ├── model.py           GlossWSDModel: encode → score → softmax, save / load
│   ├── train.py           fine-tuning loop with validation and early stopping
│   ├── evaluate.py        test metrics, predictions, confusion matrix
│   ├── baseline_svm.py    TF-IDF + Linear SVM baseline for comparison
│   └── visualize_stats.py dataset plots
├── backend/app.py         Flask API: /predict, /metadata, /examples
├── web/                   browser interface
├── notebooks/ArthoBodh_WSD_Colab.ipynb   train + evaluate on a free Colab GPU
├── docs/                  detailed explanations (see below)
├── checkpoints/           fine-tuned model (created by training, not in git)
├── results/               metrics and plots
├── server.py / run_server.bat
└── requirements.txt
```

---

## 4. Running it

### Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Train (GPU strongly recommended)
Full training takes a few minutes per epoch on a Colab T4 GPU, but hours on a laptop CPU.

- **Colab:** upload `notebooks/ArthoBodh_WSD_Colab.ipynb` to Colab, set Runtime → T4 GPU, and run all cells. When asked,
  upload **only** `data/processed/dataset_splits.json`; the notebook already contains the model code (an exact copy of `src/`).
  It trains, evaluates, and gives you `arthobodh_model.zip` (download or Google Drive); unzip it inside `ArthoBodh/`.
  If you change anything in `src/`, the notebook's code cells must be updated to match.
- **Local GPU:**
  ```bash
  python -m src.train
  python -m src.evaluate
  ```
- Quick check that everything runs (CPU, about 2 minutes, not a usable model):
  `python -m src.train --limit 16 --epochs 1 --output checkpoints/smoke`

### Use the web app
```bash
run_server.bat        # or: python server.py
```
Open http://127.0.0.1:8000, choose an example (held-out test sentences) or type your own sentence with one of the
100 supported words. Without a trained checkpoint the page tells you so instead of guessing.

### Baseline and plots
```bash
python -m src.baseline_svm
python -m src.visualize_stats
```

---

## 5. Results (test set, 690 paragraphs)

| Model | Accuracy | Macro F1 | 3-sense words | 4-sense words |
|---|:-:|:-:|:-:|:-:|
| TF-IDF + Linear SVM (one model per word) | 69.42% | 69.53% | 70.30% | 68.61% |
| **BanglaBERT gloss cross-encoder** | **94.78%** | **94.89%** | 93.64% | 95.83% |

`src.evaluate` also writes per-word accuracy (the hardest words are listed first) and every test prediction to
`results/metrics/transformer_test_predictions.json` for error analysis.

---

## 6. Further reading

- [docs/how_the_model_works.md](docs/how_the_model_works.md): one real sentence traced through every step, with the actual tokens, tensor shapes, scores and probabilities.
- [docs/methods_and_future_work.md](docs/methods_and_future_work.md): how ArthoBodh relates to knowledge-based, supervised and unsupervised WSD, and directions for further work.
- [data/supported_words.txt](data/supported_words.txt): the 100 target words and their 345 meanings.
