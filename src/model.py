"""
Gloss-matching cross-encoder for Bengali WSD.

    pair (marked context, "word : definition")
        -> BanglaBERT encoder
        -> [CLS] vector
        -> small classification head
        -> one real-valued match score

All candidate senses of a word are scored this way, the scores are put side by side,
and a softmax over them gives the probability of each sense. Training minimises
cross-entropy against the correct sense, which pushes the right definition's score up
and the other definitions' scores down for that context.

Because the definition text is part of the input, the model learns what it means for
a context to *match a meaning*, instead of memorising "slot 2 of word 57".
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .text import build_pairs


class GlossWSDModel:
    def __init__(self, encoder, tokenizer, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.tokenizer = tokenizer

    # ---- construction / persistence -------------------------------------------------

    @classmethod
    def from_pretrained_base(cls, base_model=config.BASE_MODEL, device=None):
        """Fresh model for training: pretrained BanglaBERT + untrained 1-output scoring head."""
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        encoder = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
        return cls(encoder, tokenizer, device)

    @classmethod
    def load(cls, checkpoint_dir=config.CHECKPOINT_DIR, device=None):
        """Fine-tuned model saved by save()."""
        checkpoint_dir = Path(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        encoder = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        model = cls(encoder, tokenizer, device)
        model.encoder.eval()
        return model

    def save(self, checkpoint_dir=config.CHECKPOINT_DIR, info=None):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        if info is not None:
            with open(checkpoint_dir / "training_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

    # ---- scoring --------------------------------------------------------------------

    def encode(self, items):
        """
        items: list of (context, target_word, senses_dict).
        Returns tokenized pairs for every candidate of every item, plus where each pair
        belongs in the [num_items, max_senses] score matrix.
        """
        firsts, seconds, rows, cols, sense_nums = [], [], [], [], []
        for row, (context, target, senses) in enumerate(items):
            f, s, nums, _ = build_pairs(context, target, senses)
            firsts += f
            seconds += s
            rows += [row] * len(nums)
            cols += list(range(len(nums)))
            sense_nums.append(nums)

        enc = self.tokenizer(
            firsts, seconds,
            max_length=config.MAX_LEN,
            truncation="only_first",       # never cut the definition
            padding=True,
            return_tensors="pt",
        )
        return enc, torch.tensor(rows), torch.tensor(cols), sense_nums

    def score(self, enc, rows, cols, num_items):
        """Forward pass -> [num_items, max_senses] logits; missing senses are -inf."""
        enc = {k: v.to(self.device) for k, v in enc.items()}
        pair_scores = self.encoder(**enc).logits.squeeze(-1).float()
        max_senses = int(cols.max().item()) + 1
        matrix = torch.full((num_items, max_senses), float("-inf"), device=self.device)
        matrix[rows.to(self.device), cols.to(self.device)] = pair_scores
        return matrix

    @torch.no_grad()
    def predict(self, context, target_word, senses):
        """
        Returns a list of {"sense_num", "sense", "probability"} sorted by probability,
        plus whether the target word was found in the context.
        """
        self.encoder.eval()
        _, _, _, found = build_pairs(context, target_word, senses)
        enc, rows, cols, sense_nums = self.encode([(context, target_word, senses)])
        probs = torch.softmax(self.score(enc, rows, cols, 1), dim=-1)[0].cpu().tolist()
        defs = {int(k): v for k, v in senses.items()}
        ranked = [
            {"sense_num": num, "sense": defs[num], "probability": probs[i]}
            for i, num in enumerate(sense_nums[0])
        ]
        ranked.sort(key=lambda r: -r["probability"])
        return ranked, found
