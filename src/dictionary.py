"""
src/dictionary.py
Multi-tier dictionary resolution for Bengali WSD:
1. Priority 1: Curated 100-word catalog from data/processed/dataset_splits.json.
2. Priority 2: Concise everyday supplementary lexicon from data/supplementary_lexicon.json.
3. Priority 3: Bengali IndoWordNet (pyiwn), cleaned by src/wordnet_senses.py.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from . import wordnet_senses
from .text import normalize_text


class BengaliDictionary:
    def __init__(self, catalog_senses: Optional[dict] = None):
        """
        catalog_senses: dict mapping normalized target_word -> {sense_num: definition}
        """
        self.catalog = catalog_senses or {}
        self.supplementary = self._load_supplementary_lexicon()

    def _load_supplementary_lexicon(self) -> dict:
        lex_path = Path(__file__).resolve().parent.parent / "data" / "supplementary_lexicon.json"
        if lex_path.exists():
            try:
                with open(lex_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {normalize_text(k): v for k, v in data.items()}
            except Exception as e:
                print(f"[Warning] Failed to load supplementary lexicon: {e}", file=sys.stderr)
        return {}

    def wordnet_available(self) -> bool:
        """Loads IndoWordNet on first call; False if it could not be loaded."""
        return wordnet_senses.load_wordnet() is not None

    def get_senses(self, word: str, pos: Optional[str] = None) -> dict:
        """
        Resolves candidate definitions for the target word. pos ("noun", "verb",
        "adjective", "adverb") only filters IndoWordNet senses; the catalog and
        supplementary lexicon have no part-of-speech information.

        Returns:
            {
                "senses": {1: "definition 1", 2: "definition 2", ...},
                "source": "catalog" | "supplementary" | "indowordnet" | "not_found",
                "is_monosemous": bool
            }
        """
        word = normalize_text(word)

        # 1. First priority: Check project curated 100-word catalog
        if word in self.catalog and self.catalog[word]:
            senses = {int(k): normalize_text(v) for k, v in self.catalog[word].items()}
            return {
                "senses": senses,
                "source": "catalog",
                "is_monosemous": len(senses) == 1,
            }

        # 2. Second priority: Check concise everyday supplementary lexicon
        if word in self.supplementary and self.supplementary[word]:
            senses = {int(k): normalize_text(v) for k, v in self.supplementary[word].items()}
            return {
                "senses": senses,
                "source": "supplementary",
                "is_monosemous": len(senses) == 1,
            }

        # 3. Third priority: Bengali IndoWordNet, cleaned into short training-style definitions
        senses = wordnet_senses.get_senses(word, pos=pos)
        if senses:
            return {
                "senses": senses,
                "source": "indowordnet",
                "is_monosemous": len(senses) == 1,
            }

        # 4. Not found in any dictionary
        return {
            "senses": {},
            "source": "not_found",
            "is_monosemous": False,
        }
