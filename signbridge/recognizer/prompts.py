"""VLM prompt builders for SignBridge.

Two builders share a common closed-vocabulary block and a common output
contract ("ONE token only, no explanation"). Single-frame and multi-frame
prompts diverge only in the framing — multi-frame adds explicit frame
markers and a 'motion across frames' priming line.

Patterns from NVIDIA's Vision-Language-Model Prompt Engineering Guide
(March 2025): closed-vocabulary forcing, domain priming preface,
sequential frame markers for temporal content.
"""

from __future__ import annotations

from signbridge.vocab import VOCAB_PROMPT_LITERAL

_DOMAIN_PRIME = (
    "American Sign Language (ASL) is a visual language with one-handed "
    "and two-handed signs. Fingerspelled letters (A-Z) and numbers (0-9) "
    "are typically held STATIC for ~0.5 seconds. Lexical signs (hello, "
    "thank_you, please, eat, drink, ...) are typically MOTION over ~1-2 "
    "seconds — the meaning is in the movement, not a single frame."
)

_OUTPUT_CONTRACT = (
    "Reply with EXACTLY ONE token from this list, no other text, no "
    "quotes, no explanation:\n"
    f"{VOCAB_PROMPT_LITERAL}\n\n"
    "Rules:\n"
    "- Single uppercase letter (A-Z) for fingerspelling letters.\n"
    "- Single digit (0-9) for fingerspelled numbers.\n"
    "- Lowercase word with underscores for full-word signs (e.g. thank_you).\n"
    "- 'unknown' if no sign is visible or the gesture isn't in the list.\n"
    "- Do NOT explain. Do NOT add punctuation. Single token only."
)


def build_single_frame_prompt() -> str:
    """Prompt for single-image sign recognition (fingerspelled letters etc.)."""
    return (
        f"{_DOMAIN_PRIME}\n\n"
        "Look at this image of a single signed gesture and identify which "
        "ASL sign or fingerspelled letter is shown.\n\n"
        f"{_OUTPUT_CONTRACT}"
    )


def build_multi_frame_prompt(n_frames: int) -> str:
    """Prompt for multi-image sign recognition (motion-dependent signs).

    Generates explicit 'Frame 1: ... Frame N: ...' markers so the VLM treats
    the inputs as a temporal sequence rather than independent samples.
    """
    if n_frames < 2:
        raise ValueError(
            f"multi-frame prompt requires n_frames >= 2, got {n_frames}"
        )
    frame_markers = "\n".join(
        f"- Frame {i}: image {i} of {n_frames}" for i in range(1, n_frames + 1)
    )
    return (
        f"{_DOMAIN_PRIME}\n\n"
        f"You are about to see {n_frames} images captured in temporal order, "
        "spaced evenly over a 1.5-second window. The images are frames from "
        "a single gesture. The meaning is the MOTION across the frames, not "
        "any single image.\n\n"
        f"{frame_markers}\n\n"
        "Identify the ASL sign or fingerspelled letter being made across "
        "this sequence.\n\n"
        f"{_OUTPUT_CONTRACT}"
    )
