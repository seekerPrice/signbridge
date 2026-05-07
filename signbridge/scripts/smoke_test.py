"""End-to-end smoke test for the SignBridge inference path.

Run AFTER you've filled in .env with provider credentials. Exercises:
- /info endpoint (no auth needed)
- the VLM recognizer with a synthetic frame
- the LLM composer with a hand-crafted sign sequence
- the TTS pipeline

Usage:
    python -m signbridge.scripts.smoke_test
    SIGNBRIDGE_PROVIDER=openai python -m signbridge.scripts.smoke_test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from signbridge.composer.sentence import compose_sentence
from signbridge.imageio import load_rgb
from signbridge.recognizer.vlm import recognize_sign_from_frame
from signbridge.voice.tts import synthesize_speech


def _make_synthetic_frame() -> np.ndarray:
    """Create a 256x256 RGB image with a stylised pose silhouette.

    Real recognition needs an actual hand/sign image. This synthetic frame
    is just to confirm the API plumbing works end-to-end — accuracy is
    expected to be 'unknown' (the VLM returning 'unknown' is the right
    answer for a stick figure).
    """
    img = Image.new("RGB", (256, 256), color=(245, 245, 245))
    d = ImageDraw.Draw(img)
    # Stick figure: head + body + arms in "A" sign pose
    d.ellipse((110, 30, 146, 66), fill=(220, 180, 140), outline="black", width=2)
    d.line((128, 66, 128, 160), fill="black", width=4)
    d.line((128, 90, 95, 130), fill="black", width=4)
    d.line((128, 90, 161, 130), fill="black", width=4)
    d.ellipse((85, 120, 105, 140), fill=(220, 180, 140), outline="black", width=2)
    d.ellipse((151, 120, 171, 140), fill=(220, 180, 140), outline="black", width=2)
    d.text((90, 200), "synthetic test frame", fill="black")
    return np.asarray(img)


def _print_provider_info() -> None:
    provider = os.getenv("SIGNBRIDGE_PROVIDER", "amd")
    print(f"  provider = {provider}")
    if provider == "amd":
        base = os.getenv("AMD_DEV_CLOUD_BASE_URL", "")
        key = os.getenv("AMD_DEV_CLOUD_API_KEY", "")
        print(f"  AMD_DEV_CLOUD_BASE_URL = {base or '(unset)'}")
        print(f"  AMD_DEV_CLOUD_API_KEY  = {'set (' + str(len(key)) + ' chars)' if key else '(unset)'}")
    elif provider == "openai":
        print(f"  OPENAI_API_KEY = {'set' if os.getenv('OPENAI_API_KEY') else '(unset)'}")
    elif provider == "hf":
        print(f"  HF_TOKEN = {'set' if os.getenv('HF_TOKEN') else '(unset)'}")


def _step(label: str) -> None:
    print(f"\n── {label} ──")


def main() -> int:
    parser = argparse.ArgumentParser(description="SignBridge end-to-end smoke test")
    parser.add_argument(
        "--text",
        default="My name is Lucas. Hello.",
        help="Text to synthesise via TTS",
    )
    parser.add_argument(
        "--signs",
        nargs="+",
        default=["hello", "name", "L", "U", "C", "A", "S"],
        help="Sign sequence to compose into a sentence",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        default=None,
        help="Path to a real sign image (PNG/JPG). Default = synthetic frame.",
    )
    args = parser.parse_args()

    load_dotenv()

    _step("Provider config")
    _print_provider_info()

    _step("VLM recognizer (sign-frame → token)")
    if args.frame:
        img = load_rgb(args.frame)
        print(f"  using real frame: {args.frame} ({img.shape})")
    else:
        img = _make_synthetic_frame()
        print(f"  using synthetic frame ({img.shape})")
        print("  (a synthetic stick figure is unlikely to match an ASL sign;")
        print("   the expected outcome is the VLM returning 'unknown' or empty —")
        print("   that proves the call worked even when accuracy can't be measured.)")
    t0 = time.perf_counter()
    token, conf = recognize_sign_from_frame(img)
    dt = time.perf_counter() - t0
    print(f"  → token={token!r} confidence={conf:.2f} latency={dt:.2f}s")

    _step("LLM composer (sign tokens → English sentence)")
    print(f"  input signs: {args.signs}")
    t0 = time.perf_counter()
    sentence = compose_sentence(args.signs)
    dt = time.perf_counter() - t0
    print(f"  → sentence = {sentence!r}  ({dt:.2f}s)")

    _step("TTS (text → audio)")
    print(f"  input text: {args.text!r}")
    t0 = time.perf_counter()
    audio_path = synthesize_speech(args.text)
    dt = time.perf_counter() - t0
    if audio_path:
        size = Path(audio_path).stat().st_size
        print(f"  → wrote {audio_path}  ({size:,} bytes, {dt:.2f}s)")
    else:
        print("  → no audio (TTS unavailable)")

    _step("Summary")
    ok_recognize = bool(token)
    ok_compose = bool(sentence)
    ok_tts = bool(audio_path)
    flags = {
        "recognizer": "✓" if ok_recognize else "—  (provider may be in stub mode; check creds)",
        "composer":   "✓" if ok_compose else "✗  composer failed",
        "tts":        "✓" if ok_tts else "✗  TTS failed",
    }
    for k, v in flags.items():
        print(f"  {k:<10} {v}")

    # Compose + tts MUST work even with no provider (naive joiner + silent stub).
    # Recognizer needs a real provider.
    return 0 if (ok_compose and ok_tts) else 1


if __name__ == "__main__":
    sys.exit(main())
