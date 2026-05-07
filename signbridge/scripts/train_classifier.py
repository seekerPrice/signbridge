"""Train the sign classifier on WLASL Top-100 + ASL fingerspelling alphabet.

Run on AMD Developer Cloud (Day 2). Produces:
- models/classifier.pt  — TorchScript checkpoint loaded by classifier.py
- a brief training-report.txt for the technical walkthrough

This is a Day-2 deliverable; the file is scaffolded with TODOs for the
actual training loop. The model architecture is locked: a 3-layer
transformer encoder over (T, 543) landmark sequences with a classification
head over the VOCABULARY in `signbridge.recognizer.classifier`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SignBridge classifier")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/wlasl"),
        help="Path to WLASL Top-100 dataset root (with extracted landmark .npz files).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/classifier.pt"),
        help="Where to save the trained TorchScript model.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=16,
        help="Sequence length (frames per sample).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda | cpu (on AMD Dev Cloud, ROCm exposes as cuda).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if not args.dataset.exists():
        logger.error("dataset not found at %s — download WLASL Top-100 first.", args.dataset)
        logger.error(
            "see: https://github.com/dxli94/WLASL — clone the repo, run their start_kit, "
            "then extract landmarks with `python -m signbridge.scripts.extract_landmarks`."
        )
        return 1

    # TODO(Day 2):
    #   1. Load WLASL Top-100 + ASL fingerspelling alphabet samples (pre-extracted landmark .npz).
    #   2. Train/val split 80/20 stratified by class.
    #   3. Build a 3-layer transformer encoder:
    #        - input proj 543 → 256
    #        - 3× TransformerEncoderLayer(d_model=256, nhead=4, dim_ff=512)
    #        - mean-pool over time → linear → VOCAB_SIZE
    #   4. AdamW + cosine LR + label smoothing 0.1 + dropout 0.2.
    #   5. Train args.epochs, log val accuracy each epoch.
    #   6. torch.jit.script the best-val checkpoint and save to args.output.
    #   7. Write a training-report.txt with peak val acc, per-class F1, and ROCm telemetry
    #      (peak GPU memory, throughput) for the walkthrough.
    logger.warning(
        "training loop not yet implemented — Day 2 deliverable. "
        "Args parsed OK: dataset=%s epochs=%d batch=%d lr=%s window=%d output=%s device=%s",
        args.dataset, args.epochs, args.batch_size, args.lr, args.window, args.output, args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
