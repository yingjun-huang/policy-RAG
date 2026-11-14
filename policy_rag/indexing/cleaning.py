"""Text normalization and near-duplicate detection."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Zero-width spaces and formatting artifacts
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CJK_HARD_BREAK = re.compile(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])")


def normalize_text(text: str, *, join_hard_breaks: bool = True) -> str:
    """UTF-8-safe cleanup: removes zero-width chars, applies NFKC normalization,
    strips control characters, collapses whitespace, and repairs awkward CJK line-breaks.
    """
    if not text:
        return ""
    cleaned = text.translate(_ZERO_WIDTH)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = _CONTROL.sub("", cleaned)
    if join_hard_breaks:
        cleaned = _CJK_HARD_BREAK.sub("", cleaned)
    cleaned = _INLINE_SPACE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()


def simhash64(text: str) -> int:
    """Computes 64-bit character 2-gram SimHash for near-duplicate detection."""
    if not text:
        return 0
    grams = [text[i : i + 2] for i in range(len(text) - 1)] or [text]
    weights = [0] * 64
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        val = int.from_bytes(digest[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if (val >> bit) & 1 else -1
    fingerprint = 0
    for bit, score in enumerate(weights):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming64(left: int, right: int) -> int:
    """Bit-level Hamming distance between two 64-bit fingerprints."""
    xor = (left ^ right) & ((1 << 64) - 1)
    return bin(xor).count("1")
