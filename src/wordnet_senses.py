"""
Turns Bengali IndoWordNet synsets into short sense definitions in the style of the
training catalog, so the model sees definitions it knows how to score.

    get_senses("কল")  ->  {1: "যন্ত্র, মেশিন ; সেই উপকরণ যা কোনো বিশেষ কাজ",
                           2: "মিল ; শস্য, মকাই, দানা প্রভৃতি পেষাই করার যন্ত্র",
                           3: "জলবাহিনীর একটি প্রান্ত যেটিতে মুখনল থাকে"}

pyiwn API used (checked against pyiwn 0.0.5):
    pyiwn.IndoWordNet(pyiwn.Language.BENGALI)
    iwn.synsets(word, pos=None)   pos: pyiwn.PosTag; raises KeyError for unknown words
    synset.lemma_names() -> list[str] ('_' joins multi-word lemmas)
    synset.gloss() -> str,  synset.examples() -> list[str],  synset.pos() -> str,
    synset.synset_id() -> int
"""

import builtins
import functools
import re
import sys

from .text import normalize_text

MAX_SYNONYMS = 3
MAX_GLOSS_WORDS = 8
MAX_DEFINITION_WORDS = 20

POS_TAGS = ("noun", "verb", "adjective", "adverb")

# Inflectional suffixes stripped to find a base form, longest first so কলের -> কল (not কলে).
_SUFFIXES = ("গুলোর", "গুলোকে", "গুলো", "গুলি", "দের", "েরা", "ের", "টার", "টির", "টা", "টি",
             "তে", "কে", "রা", "র", "ে", "য়", "এ")
_SUFFIXES = tuple(sorted({normalize_text(s) for s in _SUFFIXES}, key=len, reverse=True))

# Connectives that must not end a shortened gloss ("... করার বা", "... থাকে ও").
_DANGLING_WORDS = {"বা", "ও", "এবং", "আর", "যা", "যে", "যার", "যেটি", "যেটিতে", "যাতে", "যেখানে",
                   "প্রভৃতি", "ইত্যাদি", "করার", "জন্য", "সেই", "কোনো", "কোনও", "এমন"}

_QUOTES = re.compile(r"[\"“”‘’'«»`]")
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")
_EXAMPLE_SEPARATOR = re.compile(r"[:;।॥৷|]")
_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")
_NOT_BENGALI_OR_COMMA = re.compile(r"[^ঀ-৿\s,]")
_COMMAS = re.compile(r"\s*,[\s,]*")
_SPACES = re.compile(r"\s+")
_BENGALI_LETTER = re.compile(r"[অ-হৎড়-য়]")


# ---- IndoWordNet (loaded once, on first use) -----------------------------------------
_iwn = None
_iwn_error = None


def load_wordnet():
    """Returns the Bengali IndoWordNet, or None if it cannot be loaded (see wordnet_error())."""
    global _iwn, _iwn_error
    if _iwn is None and _iwn_error is None:
        try:
            import pyiwn
            import pyiwn.iwn
            # pyiwn reads its UTF-8 data files with a bare open(), which uses the Windows
            # code page and crashes. Give only the pyiwn module a UTF-8 open().
            pyiwn.iwn.open = functools.partial(builtins.open, encoding="utf-8")
            _iwn = pyiwn.IndoWordNet(pyiwn.Language.BENGALI)
        except Exception as e:
            _iwn_error = f"{type(e).__name__}: {e}"
            print(f"[warning] Bengali IndoWordNet could not be loaded: {_iwn_error}", file=sys.stderr)
    return _iwn


def wordnet_error():
    return _iwn_error


# ---- cleaning -----------------------------------------------------------------------
def clean_text(text) -> str:
    """Cleans a gloss or lemma name; returns "" if no Bengali word is left."""
    text = str(text or "").replace("_", " ")
    text = _QUOTES.sub("", text)
    previous = None
    while previous != text:                      # repeat to also remove nested brackets
        previous, text = text, _BRACKETED.sub(" ", text)
    text = _EXAMPLE_SEPARATOR.split(text, maxsplit=1)[0]
    text = _LATIN_OR_DIGIT.sub(" ", text)
    text = _NOT_BENGALI_OR_COMMA.sub(" ", text)  # stray punctuation and leftover brackets
    text = normalize_text(_SPACES.sub(" ", text))
    text = _COMMAS.sub(", ", text).strip(" ,")
    return text if _BENGALI_LETTER.search(text) else ""


def _base_forms(word: str) -> list[str]:
    """The word itself, then candidate base forms with one common suffix removed."""
    forms = [word]
    for suffix in _SUFFIXES:
        base = word[: -len(suffix)]
        if word.endswith(suffix) and len(base) >= 2 and base not in forms:
            forms.append(base)
    return forms


def _is_target_form(lemma: str, target_forms: set) -> bool:
    """True for the target itself or an inflected form of it (কল, কলের, কলটা)."""
    return lemma in target_forms or any(
        lemma == form + suffix for form in target_forms for suffix in _SUFFIXES)


def _synonyms(synset, target_forms: set) -> list[str]:
    """All cleaned lemma names except the target and its inflections, without duplicates."""
    synonyms = []
    for lemma in synset.lemma_names():
        lemma = clean_text(lemma)
        if lemma and lemma not in synonyms and not _is_target_form(lemma, target_forms):
            synonyms.append(lemma)
    return synonyms


def _short_gloss_words(gloss: str) -> list[str]:
    words = gloss.split()[:MAX_GLOSS_WORDS]
    while words and (words[-1].rstrip(",") in _DANGLING_WORDS or words[-1] == ","):
        words.pop()
    if words:
        words[-1] = words[-1].rstrip(",")
    return words


def short_definition(synset, target: str) -> str:
    """
    "syn1, syn2, syn3 ; short gloss" in the training-catalog style, at most 20 words.
    Uses only synonym lemmas and the gloss, never the synset's example sentences.
    """
    target = normalize_text(target)
    synonyms = _synonyms(synset, set(_base_forms(target)))[:MAX_SYNONYMS]
    gloss_words = _short_gloss_words(clean_text(synset.gloss()))
    if " ".join(gloss_words) in synonyms:        # gloss only repeats a synonym
        gloss_words = []

    synonym_words = sum(len(s.split()) for s in synonyms)
    if synonym_words + len(gloss_words) > MAX_DEFINITION_WORDS:
        gloss_words = _short_gloss_words(" ".join(gloss_words[: max(0, MAX_DEFINITION_WORDS - synonym_words)]))
    while synonyms and sum(len(s.split()) for s in synonyms) + len(gloss_words) > MAX_DEFINITION_WORDS:
        synonyms.pop()

    gloss = " ".join(gloss_words)
    if synonyms and gloss:
        return f"{', '.join(synonyms)} ; {gloss}"
    return ", ".join(synonyms) or gloss


# ---- sense selection ----------------------------------------------------------------
def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def find_synsets(target: str, pos: str | None = None) -> tuple[str | None, list]:
    """
    (matched_form, synsets) for the target, trying the normalized word first and then
    its base forms (কলের -> কল). pos is one of POS_TAGS or None.
    """
    if pos is not None and pos not in POS_TAGS:
        raise ValueError(f"pos must be one of {POS_TAGS}, got {pos!r}")
    iwn = load_wordnet()
    if iwn is None:
        return None, []

    import pyiwn
    pos_tag = pyiwn.PosTag(pos) if pos else None
    for form in _base_forms(normalize_text(target)):
        try:
            synsets = iwn.synsets(form, pos=pos_tag)
        except KeyError:                         # word not in IndoWordNet
            continue
        if synsets:
            return form, synsets
    return None, []


def get_senses(target: str, pos: str | None = None, max_senses: int = 4) -> dict[int, str]:
    """
    Up to max_senses distinct senses of the target from IndoWordNet, in WordNet order,
    as {1: "definition", 2: ...} (the same shape as the catalog senses in src/model.py).
    Returns {} if the word is not found or IndoWordNet is unavailable.
    """
    target = normalize_text(target)
    _, synsets = find_synsets(target, pos)
    target_forms = set(_base_forms(target))

    kept = []   # (definition, synonym set, gloss word set)
    for synset in synsets:
        definition = short_definition(synset, target)
        if not definition:
            continue
        synonyms = set(_synonyms(synset, target_forms))
        gloss_words = set(clean_text(synset.gloss()).replace(",", " ").split())
        if any(definition == d or len(synonyms & s) >= 2 or _jaccard(gloss_words, g) > 0.5
               for d, s, g in kept):
            continue
        kept.append((definition, synonyms, gloss_words))
        if len(kept) == max_senses:
            break
    return {i: definition for i, (definition, _, _) in enumerate(kept, start=1)}
