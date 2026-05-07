"""Shared closed-vocabulary constants for SignBridge.

Single source of truth so the VLM recognizer (`signbridge.recognizer.vlm`)
and the trained-classifier path (`signbridge.recognizer.classifier`) can
never drift. The classifier head must produce logits in this exact order;
the VLM prompt forces the model to choose only from this set.
"""

from __future__ import annotations

ALPHABET: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS: tuple[str, ...] = tuple("0123456789")
WLASL_TOP50: tuple[str, ...] = (
    "hello", "thank_you", "name", "please", "sorry", "yes", "no", "good",
    "bad", "help", "want", "like", "love", "family", "friend", "mother",
    "father", "sister", "brother", "child", "home", "school", "work",
    "eat", "drink", "water", "food", "more", "finish", "today", "tomorrow",
    "yesterday", "where", "what", "who", "why", "when", "how", "go", "come",
    "see", "know", "understand", "think", "feel", "happy", "sad", "tired",
    "hungry", "wait",
)

# Sentinel returned by the VLM when no recognized sign is present.
UNKNOWN: str = "unknown"

VOCAB: tuple[str, ...] = ALPHABET + DIGITS + WLASL_TOP50 + (UNKNOWN,)
VOCAB_SET: frozenset[str] = frozenset(VOCAB)

# Pre-rendered space-separated string for prompt embedding.
VOCAB_PROMPT_LITERAL: str = " ".join(VOCAB)
