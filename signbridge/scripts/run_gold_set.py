"""Accuracy harness — run the recognizer over a labelled sample folder.

Folder layout expected:
    tests/golden/
        A/<any>.jpg|png    →  expected token "A"
        B/<any>.jpg|png    →  expected token "B"
        ...
        hello/<any>.jpg|png →  expected token "hello"

Each subdirectory name is the expected token. Every image inside is a sample.

Output:
- per-class accuracy (correct / total)
- overall accuracy
- a CSV at tests/golden/results-<timestamp>.csv

Usage:
    python -m signbridge.scripts.run_gold_set
    python -m signbridge.scripts.run_gold_set --root tests/golden
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from signbridge.imageio import load_rgb
from signbridge.recognizer.vlm import recognize_sign_from_frame

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _iter_samples(root: Path):
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXTS:
                yield cls, img_path


def main() -> int:
    parser = argparse.ArgumentParser(description="SignBridge accuracy harness")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("tests/golden"),
        help="Root folder with one subdirectory per expected token",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (defaults to tests/golden/results-<ts>.csv)",
    )
    args = parser.parse_args()

    load_dotenv()

    if not args.root.exists() or not args.root.is_dir():
        print(f"error: {args.root} not found or not a directory", file=sys.stderr)
        print("create it with subdirectories named after expected tokens, e.g.:", file=sys.stderr)
        print("    tests/golden/A/sample1.jpg", file=sys.stderr)
        print("    tests/golden/hello/sample2.png", file=sys.stderr)
        return 2

    samples = list(_iter_samples(args.root))
    if not samples:
        print(f"no images found under {args.root}", file=sys.stderr)
        return 2

    out_path = args.output or args.root / f"results-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    rows: list[dict[str, str]] = []

    print(f"running {len(samples)} samples against the configured provider…")
    t_start = time.perf_counter()
    for expected, path in samples:
        per_class_total[expected] += 1
        img = load_rgb(path)
        t0 = time.perf_counter()
        predicted, confidence = recognize_sign_from_frame(img)
        dt_ms = (time.perf_counter() - t0) * 1000
        ok = predicted == expected
        if ok:
            per_class_correct[expected] += 1
        rows.append(
            {
                "path": str(path),
                "expected": expected,
                "predicted": predicted,
                "confidence": f"{confidence:.2f}",
                "latency_ms": f"{dt_ms:.0f}",
                "correct": "1" if ok else "0",
            }
        )
        print(
            f"  [{'✓' if ok else '✗'}] {expected:<10} → {predicted!r:<12} "
            f"conf={confidence:.2f}  {dt_ms:.0f}ms  ({path.name})"
        )

    total_correct = sum(per_class_correct.values())
    total = sum(per_class_total.values())
    overall = total_correct / total if total else 0.0
    elapsed = time.perf_counter() - t_start

    print()
    print("Per-class accuracy:")
    for cls in sorted(per_class_total):
        c = per_class_correct[cls]
        n = per_class_total[cls]
        print(f"  {cls:<12} {c}/{n}  ({(c / n) * 100:.0f}%)" if n else f"  {cls:<12} 0/0")
    print()
    print(f"Overall: {total_correct}/{total}  ({overall * 100:.1f}%)")
    print(f"Total wall time: {elapsed:.1f}s  (avg {(elapsed / total) * 1000:.0f}ms per sample)")

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path", "expected", "predicted", "confidence", "latency_ms", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {out_path}")

    # Exit non-zero if accuracy below the V1 success criterion (75%).
    return 0 if overall >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())
