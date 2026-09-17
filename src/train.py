"""
Fine-tunes BanglaBERT as a gloss-matching cross-encoder.

    python -m src.train                       # full training (use a GPU, e.g. Colab T4)
    python -m src.train --limit 16 --epochs 1 # quick CPU smoke test

Each epoch:
  1. For every training context, score all candidate senses of its word.
  2. Cross-entropy over those scores against the correct sense; update the model.
  3. Measure accuracy on the validation split.
  4. If validation accuracy improved, save the model to checkpoints/banglabert-wsd.
Training stops early after PATIENCE epochs without improvement.
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from . import config
from .evaluate import label_of, load_splits, run_inference
from .model import GlossWSDModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--output", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="use only N train/val records (smoke test)")
    args = parser.parse_args()

    set_seed(config.SEED)
    data = load_splits()
    catalog = data["catalog"]
    train, val = data["train"], data["val"]
    if args.limit:
        train, val = train[:args.limit], val[:args.limit]

    model = GlossWSDModel.from_pretrained_base()
    device = model.device
    use_amp = device.type == "cuda"

    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Base model: {config.BASE_MODEL} | device: {device} | mixed precision: {use_amp}")
    print(f"Train contexts: {len(train)} | Val contexts: {len(val)} | "
          f"batch: {args.batch_size} | lr: {args.lr} | max epochs: {args.epochs}")

    best_acc, best_loss, bad_epochs, history = -1.0, float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.encoder.train()
        order = list(range(len(train)))
        random.shuffle(order)
        running_loss, correct = 0.0, 0

        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            batch = [train[i] for i in order[start:start + args.batch_size]]
            items = [(r["text"], r["target_word"], catalog[r["folder"]]["senses"]) for r in batch]
            labels = torch.tensor([label_of(r, catalog) for r in batch], device=device)
            enc, rows, cols, _ = model.encode(items)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model.score(enc, rows, cols, len(batch))   # [batch, max_senses]
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * len(batch)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running_loss / (start + len(batch)):.4f}")

        train_loss, train_acc = running_loss / len(train), correct / len(train)
        val_preds, _, val_loss = run_inference(model, val, catalog)
        val_acc = float(np.mean([p == label_of(r, catalog) for p, r in zip(val_preds, val)]))
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch} ({time.time() - t0:.0f}s) | train loss {train_loss:.4f} acc {train_acc * 100:.2f}% "
              f"| val loss {val_loss:.4f} acc {val_acc * 100:.2f}%")

        # Best = highest validation accuracy; ties broken by lower validation loss.
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, bad_epochs = val_acc, val_loss, 0
            model.save(args.output, info={"base_model": config.BASE_MODEL, "best_epoch": epoch,
                                          "val_accuracy": val_acc, "val_loss": val_loss,
                                          "max_len": config.MAX_LEN, "history": history})
            print(f"  -> saved best model to {args.output}")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print(f"No validation improvement for {config.PATIENCE} epochs; stopping.")
                break

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "transformer_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation accuracy: {best_acc * 100:.2f}%. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
