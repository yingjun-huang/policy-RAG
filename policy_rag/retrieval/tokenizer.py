"""Tokenizer using jieba for Chinese lexical retrieval."""

from __future__ import annotations

import jieba


def tokenize(text: str) -> list[str]:
    """Tokenizes text using jieba, trimming whitespace and blank tokens."""
    if not text:
        return []
    # cut_for_search generates finer tokens suitable for exact match and search indexing
    tokens = [tok.strip().lower() for tok in jieba.cut_for_search(text)]
    return [tok for tok in tokens if tok]
