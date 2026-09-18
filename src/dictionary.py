"""
src/dictionary.py
Multi-tier dictionary resolution for Bengali WSD:
1. Priority 1: Curated 100-word catalog from data/processed/dataset_splits.json.
2. Priority 2: Concise everyday supplementary lexicon from data/supplementary_lexicon.json.
3. Priority 3: Cleaned Bengali IndoWordNet (pyiwn) for arbitrary vocabulary.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 environment for pyiwn and Windows console
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from .text import normalize_text

_SPACES = re.compile(r"\s+")
_OBSCURE_TERMS = ['অসুর', 'বিরাটের পুত্র', 'বৈদীক', 'কাব্যালঙ্কার', 'ইত্যাদির সেই']
_BOILERPLATE_PREFIXES = [
    'কোনো কার্যের শেষে তার ',
    'সেই প্রধান ',
    'কোনো এমন বস্তু যা ',
    'যার সহায়তায় ',
]


class BengaliDictionary:
    def __init__(self, catalog_senses: Optional[dict] = None):
        """
        catalog_senses: dict mapping normalized target_word -> {sense_num: definition}
        """
        self.catalog = catalog_senses or {}
        self.supplementary = self._load_supplementary_lexicon()
        self._iwn = None
        self._iwn_loaded = False

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

    def _get_iwn(self):
        """Lazy load IndoWordNet only when an untrained word is requested."""
        if not self._iwn_loaded:
            self._iwn_loaded = True
            try:
                import pyiwn
                self._iwn = pyiwn.IndoWordNet(pyiwn.Language.BENGALI)
            except Exception as e:
                print(f"[Warning] Bengali IndoWordNet could not be initialized: {e}", file=sys.stderr)
                self._iwn = None
        return self._iwn

    def _clean_iwn_gloss(self, gloss: str) -> str:
        """Strip boilerplate phrasing and shorten verbose definitions to the core head phrase."""
        gloss = normalize_text(gloss)
        for prefix in _BOILERPLATE_PREFIXES:
            gloss = gloss.replace(prefix, "")

        words = gloss.split()
        # Keep definition concise (up to 8 words) to match model training distribution
        if len(words) > 8:
            gloss = " ".join(words[:8])
        return gloss.strip()

    def get_senses(self, word: str) -> dict:
        """
        Resolves candidate definitions for the target word.

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

        # 3. Third priority: Query cleaned Bengali IndoWordNet
        iwn = self._get_iwn()
        if iwn is not None:
            try:
                synsets = iwn.synsets(word)
                if synsets:
                    senses = {}
                    seen_glosses = set()
                    sense_idx = 1
                    for s in synsets:
                        raw_gloss = normalize_text(s.gloss())
                        # Skip obscure mythological / archaic synsets
                        if any(term in raw_gloss for term in _OBSCURE_TERMS):
                            continue

                        # Clean and shorten gloss
                        clean_gloss = self._clean_iwn_gloss(raw_gloss)

                        # Extract synonyms from synset lemmas (excluding the target word itself)
                        lemmas = [l.replace('_', ' ') for l in s.lemma_names() if l.strip().lower() != word.strip().lower()]
                        if lemmas:
                            formatted = f"{', '.join(lemmas[:2])} — {clean_gloss}"
                        else:
                            formatted = clean_gloss

                        if formatted and formatted not in seen_glosses:
                            seen_glosses.add(formatted)
                            senses[sense_idx] = formatted
                            sense_idx += 1

                    if senses:
                        return {
                            "senses": senses,
                            "source": "indowordnet",
                            "is_monosemous": len(senses) == 1,
                        }
            except KeyError:
                # Word is not present in IndoWordNet vocabulary
                pass
            except Exception as e:
                print(f"[Warning] Error querying IndoWordNet for '{word}': {e}", file=sys.stderr)

        # 4. Not found in any dictionary
        return {
            "senses": {},
            "source": "not_found",
            "is_monosemous": False,
        }
