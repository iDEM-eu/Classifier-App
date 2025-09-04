# utils.py
from __future__ import annotations

import string
from functools import lru_cache
from typing import Set

import nltk
from langdetect import detect, LangDetectException, DetectorFactory
from nltk.corpus import stopwords

# Make langdetect deterministic
DetectorFactory.seed = 0

# Map language codes -> NLTK stopword language names
# (If a language isn't available in your NLTK stopwords corpus, we'll fall back to English.)
LANG_CODE_TO_NAME = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "ru": "russian",
    "ca": "catalan",  # may not exist in some NLTK distributions; handled by fallback
}

PUNCTUATION: Set[str] = set(string.punctuation)


def _ensure_nltk_stopwords() -> None:
    """Ensure NLTK stopwords are available (download once if missing)."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


@lru_cache(maxsize=None)
def _stopwords_for(lang_name: str) -> Set[str]:
    """
    Return stopwords for a given NLTK language name.
    If the language isn't available, return an empty set (caller can fallback).
    """
    _ensure_nltk_stopwords()
    try:
        return set(stopwords.words(lang_name))
    except (LookupError, OSError):
        # Language not present in your stopwords corpus
        return set()


def detect_language(text: str) -> str:
    """
    Detect the text language and return an NLTK stopwords language name.
    Falls back to 'english' on failure or unknown codes.
    """
    try:
        code = detect(text)
        return LANG_CODE_TO_NAME.get(code, "english")
    except LangDetectException:
        return "english"


def get_stopwords_for_text(text: str) -> Set[str]:
    """
    Convenience: detect language, return corresponding stopwords set.
    If that set is empty (language unavailable), fall back to English.
    """
    lang_name = detect_language(text)
    sw = _stopwords_for(lang_name)
    if not sw and lang_name != "english":
        sw = _stopwords_for("english")
    return sw
