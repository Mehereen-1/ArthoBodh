# WSD Methods: Where ArthoBodh Fits and What to Explore Next

## Background: three families of WSD methods

| Family | Core idea | Needs labelled data? |
|---|---|:-:|
| **Knowledge-based** (e.g. Lesk) | Compare context words with dictionary definitions of each candidate sense; pick the sense with the most overlap | No |
| **Supervised** | Treat WSD as classification; train a model on sense-annotated examples | Yes |
| **Unsupervised** | Words in similar contexts have similar meanings; cluster occurrences by context and treat clusters as senses | No |

## Which method ArthoBodh follows

**ArthoBodh is primarily a supervised system, but it borrows the central idea of knowledge-based methods, and it
builds on unsupervised pretraining.**

| Family | Where it appears in ArthoBodh |
|---|---|
| **Supervised** (main) | The model is fine-tuned on **2416 paragraphs, each labelled with its correct meaning**, by minimizing cross-entropy against those labels. Its 94.78% test accuracy depends on this labelled data. The TF-IDF + Linear SVM baseline (69.42%) is a classic supervised system: features, then a classifier. |
| **Knowledge-based** (Lesk idea) | Like Lesk, the model **compares the context with the dictionary definition of each candidate meaning and picks the best match**. The difference is how the comparison is made: Lesk counts shared words, while ArthoBodh uses BanglaBERT to *learn* the match. It can therefore match চোখে (eyes) with অশ্রু (tears) although the two share no words. This design is known as **gloss-informed supervised WSD** (GlossBERT, Huang et al. 2019). |
| **Unsupervised** (underneath) | BanglaBERT was pretrained on a large amount of **unlabelled Bengali text**, learning from context patterns, which is the same distributional principle unsupervised WSD relies on. ArthoBodh does no clustering itself, but the model's general knowledge of Bengali comes from this pretraining. |

**Summary sentence:**

> ArthoBodh follows a supervised approach using a gloss-informed cross-encoder: a pretrained BanglaBERT model (learned
> from unlabelled text) is fine-tuned on sense-annotated data to score how well each dictionary definition (as in
> knowledge-based Lesk methods) matches the context.

### Current results (test set, 690 paragraphs)

| Model | Family | Accuracy | Macro F1 | 3-sense words | 4-sense words |
|---|---|:-:|:-:|:-:|:-:|
| TF-IDF + Linear SVM | Supervised (classic) | 69.42% | 69.53% | 70.30% | 68.61% |
| BanglaBERT gloss cross-encoder | Supervised + knowledge | **94.78%** | **94.89%** | 93.64% | 95.83% |

Weakest words for the BanglaBERT model: অন্তর (50%), খোলা (62.5%), স্পষ্ট, জমি, কুঁজো (67%), মুখ (75%).
Each word has only 6–8 test paragraphs, so a single mistake changes its score by 12–17 points.

---

## Directions to explore

### A. Knowledge-based (no training data)

1. **Simplified Lesk baseline.** Score each meaning by the number of words shared between the context and the definition,
   with no training. It is quick to build and adds a knowledge-based row to the comparison. Low accuracy is expected
   because definitions in this dataset are only 1–4 words long; that result itself shows why learned matching is needed.
2. **Extended Lesk with Bengali WordNet.** Expand each definition with synonyms and related words from Bangla WordNet
   (IndoWordNet) before matching, to test whether richer dictionary knowledge helps.
3. **Richer definitions for the trained model.** Short definitions such as `ঘন: মেঘ · নিবিড় · ত্রিমাত্রিক` give the model
   little to work with. Adding an example phrase or synonyms to each definition could help the weakest words.

### B. Supervised (improve the current model)

4. **Evaluate on unseen words.** Train on 80 words and test on the remaining 20. This measures whether the model has
   learned to *use definitions* rather than memorizing the 100 training words. It would be the most valuable research
   result to add.
5. **Evaluate on single sentences.** Training used whole paragraphs (median 49 words), while users type one short
   sentence. Evaluate by cutting test paragraphs down to the sentence containing the word; if accuracy drops, train with
   sentence crops too.
6. **Make the numbers reliable.** Train with 3–5 random seeds and report mean ± standard deviation.
7. **Try other architectures.**
   - A **bi-encoder** (BEM, Blevins & Zettlemoyer 2020) encodes context and definitions separately, so definitions are
     encoded once and prediction is faster.
   - Compare with **XLM-RoBERTa** or **MuRIL** to measure the benefit of a Bengali-specific model.
8. **LLM comparison.** Give a large language model the sentence and the definitions with no training (zero-shot) and
   compare its accuracy with the fine-tuned model.

### C. Unsupervised and semi-supervised

9. **Word sense induction.** For each word, take BanglaBERT's contextual vector of the target word in every paragraph,
   cluster the vectors without labels (k-means with k = number of meanings), and compare the clusters with the true
   meanings using purity or the Adjusted Rand Index.
10. **Nearest-neighbour baseline (few-shot).** Assign a new sentence the meaning of its most similar training paragraph,
    using the same contextual vectors. It is cheap and bridges unsupervised and supervised approaches.
11. **Self-training.** Run the model on unlabelled Bengali text (Wikipedia or news), keep predictions with very high
    confidence (for example above 99%) as new training data, and retrain. This addresses the annotation bottleneck of
    supervised methods.

---

## Recommended order

1. **Lesk baseline and a clustering or nearest-neighbour baseline.** Both are quick, run on CPU, and complete a
   comparison across all three method families:

   | Approach | Method | Accuracy |
   |---|---|:-:|
   | Knowledge-based | Simplified Lesk | to be measured |
   | Unsupervised | Embedding clustering / kNN | to be measured |
   | Supervised (classic) | TF-IDF + SVM | 69.42% |
   | Supervised + knowledge | BanglaBERT gloss cross-encoder | **94.78%** |

2. **Unseen-word evaluation.** The strongest research contribution.
3. **Single-sentence evaluation and multiple seeds.** These make the demo and the reported numbers more trustworthy.

## References

- Lesk, M. (1986). *Automatic sense disambiguation using machine readable dictionaries.* SIGDOC.
- Huang, L., Sun, C., Qiu, X., Huang, X. (2019). *GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge.* EMNLP.
- Blevins, T., Zettlemoyer, L. (2020). *Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders.* ACL.
- Bhattacharjee, A. et al. (2022). *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.* Findings of NAACL.
