"""
Evaluates a fine-tuned checkpoint on the held-out test split.

    python -m src.evaluate                    # uses checkpoints/banglabert-wsd
    python -m src.evaluate --checkpoint PATH

Writes results/metrics/transformer_metrics.json, transformer_test_predictions.json and
results/plots/transformer_confusion_matrix.png. Metrics are computed on the relative
sense label (sense 1..4), the same way as the TF-IDF + SVM baseline, so the numbers
are directly comparable.
"""

import argparse
import json
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_splits():
    with open(config.SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def label_of(record, catalog):
    """Index of the record's true sense among its word's senses (sorted by sense number)."""
    nums = sorted(int(k) for k in catalog[record["folder"]]["senses"])
    return nums.index(record["sense_num"])


@torch.no_grad()
def run_inference(model, records, catalog, batch_size=config.EVAL_BATCH_SIZE):
    """Returns (predicted label indices, probability rows, mean loss)."""
    model.encoder.eval()
    preds, prob_rows, total_loss = [], [], 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        items = [(r["text"], r["target_word"], catalog[r["folder"]]["senses"]) for r in batch]
        labels = torch.tensor([label_of(r, catalog) for r in batch], device=model.device)
        enc, rows, cols, _ = model.encode(items)
        logits = model.score(enc, rows, cols, len(batch))
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        probs = torch.softmax(logits, dim=-1)
        preds += probs.argmax(dim=-1).tolist()
        prob_rows += probs.cpu().tolist()
    return preds, prob_rows, total_loss / max(1, len(records))


def compute_metrics(y_true, y_pred, num_senses):
    def block(t, p):
        if not t:
            return None
        _, _, f1_macro, _ = precision_recall_fscore_support(t, p, average="macro", zero_division=0)
        return {"instances": len(t), "accuracy": accuracy_score(t, p), "macro_f1": f1_macro}

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    three = [(t, p) for t, p, n in zip(y_true, y_pred, num_senses) if n == 3]
    four = [(t, p) for t, p, n in zip(y_true, y_pred, num_senses) if n == 4]
    return {
        "test_instances": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "weighted_f1": f1_w,
        "three_senses": block([t for t, _ in three], [p for _, p in three]),
        "four_senses": block([t for t, _ in four], [p for _, p in four]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def save_confusion_matrix(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, cmap=plt.cm.Greens)
    fig.colorbar(im, ax=ax)
    names = ["Sense 1", "Sense 2", "Sense 3", "Sense 4"]
    ax.set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names,
           title="BanglaBERT gloss cross-encoder: confusion matrix",
           ylabel="True sense", xlabel="Predicted sense")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    from .model import GlossWSDModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test records (smoke test)")
    args = parser.parse_args()

    data = load_splits()
    catalog, test = data["catalog"], data["test"][:args.limit]

    model = GlossWSDModel.load(args.checkpoint)
    print(f"Evaluating {args.checkpoint} on {len(test)} test contexts ({model.device})...")
    preds, probs, loss = run_inference(model, test, catalog)

    y_true = [label_of(r, catalog) for r in test]
    num_senses = [len(catalog[r["folder"]]["senses"]) for r in test]
    metrics = {"model": "BanglaBERT gloss cross-encoder", "test_loss": loss,
               **compute_metrics(y_true, preds, num_senses)}

    per_word = defaultdict(lambda: [0, 0])
    predictions = []
    for r, t, p, pr in zip(test, y_true, preds, probs):
        senses = catalog[r["folder"]]["senses"]
        nums = sorted(int(k) for k in senses)
        per_word[r["target_word"]][0] += int(t == p)
        per_word[r["target_word"]][1] += 1
        predictions.append({
            "folder": r["folder"], "target_word": r["target_word"], "context": r["text"],
            "true_sense_num": nums[t], "true_sense": senses[str(nums[t])],
            "predicted_sense_num": nums[p], "predicted_sense": senses[str(nums[p])],
            "confidence": pr[p], "correct": t == p,
        })
    metrics["per_word_accuracy"] = {w: c / n for w, (c, n) in sorted(per_word.items(), key=lambda kv: kv[1][0] / kv[1][1])}

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(config.METRICS_DIR / "transformer_test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(metrics["confusion_matrix"], config.PLOTS_DIR / "transformer_confusion_matrix.png")

    print("\n================ TEST RESULTS ================")
    print(f"Accuracy:     {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:     {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1:  {metrics['weighted_f1'] * 100:.2f}%")
    for key, label in (("three_senses", "3-sense words"), ("four_senses", "4-sense words")):
        if metrics[key]:
            print(f"  {label}: accuracy {metrics[key]['accuracy'] * 100:.2f}% (n={metrics[key]['instances']})")
    baseline_path = config.METRICS_DIR / "baseline_svm_metrics.json"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)
        print(f"TF-IDF + SVM baseline accuracy: {base['accuracy'] * 100:.2f}%")
    hardest = list(metrics["per_word_accuracy"].items())[:5]
    print("Hardest words:", ", ".join(f"{w} {a * 100:.0f}%" for w, a in hardest))
    print("==============================================")
    print(f"Saved metrics and predictions to {config.METRICS_DIR}")


if __name__ == "__main__":
    main()
