# ArthoBodh (অর্থবোধ)
### A Bengali Word Sense Disambiguation (WSD) Benchmark & AI System

**ArthoBodh** is an end-to-end Bengali Word Sense Disambiguation (WSD) research framework and interactive web application. Given a polysemous Bengali target word in a sentence context, ArthoBodh resolves the exact context-dependent sense definition:

$$\text{Context } C + \text{Target Word } w \longrightarrow \text{Sense } s \in S(w)$$

---

## 📁 Repository Structure

```
ArthoBodh/
├── data/
│   ├── raw/
│   │   └── Bengali_WSD_Database/      # 100 polysemous words raw dataset (345 senses)
│   ├── processed/
│   │   └── dataset_splits.json        # Stratified Train / Val / Test partitions
│   └── prototype/                     # 3-word prototype sample (data.csv, senses.json)
│
├── src/                               # Core Python Package
│   ├── __init__.py
│   ├── data_loader.py                 # Ingestion, tag parsing & split generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_svm.py            # TF-IDF + Linear SVM baseline (69.42% accuracy)
│   │   └── transformer_wsd.py         # BanglaBERT sequence classification pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_and_analyze.py    # Error analysis & per-word breakdown
│   │   └── visualize_stats.py         # EDA distribution charts generator
│   └── prototype_arms/                # Multi-arm comparative prototypes (TF-IDF, W2V, BERT)
│       ├── __init__.py
│       ├── common.py
│       ├── tfidf_arm.py
│       ├── w2v_arm.py
│       └── bert_arm.py
│
├── backend/                           # Backend API Server
│   ├── __init__.py
│   └── app.py                        # Unified Flask server serving /predict & static web/
│
├── web/                               # Web Frontend
│   ├── index.html                     # Interactive UI
│   ├── styles.css                     # Modern stylesheet
│   ├── app.js                         # Dynamic predictions, meters & highlighting
│   └── api-config.js                  # Frontend API configuration
│
├── notebooks/                         # Google Colab & Jupyter Notebooks
│   ├── ArthoBodh_WSD_Colab.ipynb      # Complete 10-stage GPU-ready training notebook
│   └── generate_colab_notebook.py     # Script to regenerate the Colab notebook
│
├── results/                           # Evaluation Metrics, Plots & Outputs
│   ├── metrics/                       # baseline_svm_metrics.json, baseline_errors.json
│   ├── plots/                         # EDA & confusion matrix figures
│   └── prototype/                     # Prototype CSV results (bert.csv, tfidf.csv)
│
├── server.py                          # Root launcher forwarding to backend.app
├── run_server.bat                     # Windows one-click server launcher
├── requirements.txt                   # Project dependencies (UTF-8)
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
# Activate existing virtual environment (if available)
.venv\Scripts\activate

# Or install dependencies
pip install -r requirements.txt
```

### 2. Dataset Processing
To parse the raw dataset and regenerate stratified Train/Validation/Test splits:
```bash
python src/data_loader.py
```

### 3. Exploratory Data Analysis & Visualizations
Generate dataset complexity and context length plots into `results/plots/`:
```bash
python src/evaluation/visualize_stats.py
```

### 4. Classical ML Baseline (TF-IDF + Linear SVM)
Run the baseline evaluation across all 100 words and export performance metrics:
```bash
python src/models/baseline_svm.py
```
To perform in-depth misclassification analysis:
```bash
python src/evaluation/evaluate_and_analyze.py
```

### 5. Transformer Model Training (Google Colab GPU)
Because fine-tuning `csebuetnlp/banglabert` requires GPU acceleration:
1. Open [`notebooks/ArthoBodh_WSD_Colab.ipynb`](notebooks/ArthoBodh_WSD_Colab.ipynb) in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells to fine-tune the model, evaluate complexity breakdowns, and export `best_banglabert_wsd.pt`.
4. Place the downloaded `best_banglabert_wsd.pt` into the project root.

### 6. Running the Interactive Web Application
Launch the unified server using the Windows batch launcher or via Python:
```bash
run_server.bat
```
or
```bash
python server.py
```
Open `http://127.0.0.1:8000` in your browser.

---

## 📊 Baseline Results

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Linear SVM** | **69.42%** | 69.31% | 69.81% | **69.53%** | 69.42% |
| - 3-Sense Polysemous Words | 70.30% | - | - | 70.31% | 70.31% |
| - 4-Sense Polysemous Words | 68.61% | - | - | 68.65% | 68.65% |

---

## 👥 Contributors
- **ArthoBodh NLP Team**
