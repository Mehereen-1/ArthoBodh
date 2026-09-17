"""
Turns (context, target word, candidate senses) into model inputs.

Training, evaluation and the web server all go through build_pairs(), so the model
sees exactly the same input format everywhere.

For one context and a word with N senses we create N text pairs:

    first  segment:  the context, with the target word wrapped in " quotes "
    second segment:  "<target word> : <sense definition>"

BanglaBERT reads each pair jointly and scores how well that definition fits the
marked word in that context.
"""

import re

try:
    from normalizer import normalize as _bn_normalize   # csebuetnlp normalizer used when BanglaBERT was pretrained
except ImportError:                                      # pragma: no cover
    _bn_normalize = None

_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_SPACES = re.compile(r"\s+")
_BENGALI_CHAR = r"ঀ-৿"


def normalize_text(text: str) -> str:
    text = str(text)
    if _bn_normalize is not None:
        text = _bn_normalize(text)
    text = _ZERO_WIDTH.sub("", text)
    return _SPACES.sub(" ", text).strip()


def mark_target(context: str, target: str) -> tuple[str, bool]:
    """
    Wraps every occurrence of the target that starts a word (so জল matches জল and জলের,
    but not the middle of another word) in quotes. Returns (marked_text, found).
    """
    pattern = re.compile(rf"(?<![{_BENGALI_CHAR}])({re.escape(target)}[{_BENGALI_CHAR}]*)")
    marked, count = pattern.subn(r'" \1 "', context)
    return marked, count > 0


def sorted_senses(senses: dict) -> list[tuple[int, str]]:
    """Catalog senses as [(sense_num, definition), ...] ordered by sense number."""
    return sorted(((int(k), v) for k, v in senses.items()), key=lambda kv: kv[0])


def build_pairs(context: str, target: str, senses: dict):
    """
    Returns (first_segments, second_segments, sense_nums, target_found).
    One entry per candidate sense, in sense-number order.
    """
    target = normalize_text(target)
    marked, found = mark_target(normalize_text(context), target)
    firsts, seconds, nums = [], [], []
    for num, definition in sorted_senses(senses):
        firsts.append(marked)
        seconds.append(f"{target} : {normalize_text(definition)}")
        nums.append(num)
    return firsts, seconds, nums, found
