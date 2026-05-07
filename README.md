---
title: SignBridge
emoji: 🤟
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Real-time ASL → English speech on AMD MI300X.
---

# SignBridge — real-time ASL → speech

Two people who couldn't communicate, now can.

A deaf person signs into the webcam. SignBridge — a multi-stage vision + reasoning + voice pipeline running on a single AMD Instinct MI300X — translates the signs into spoken English in under 2 seconds.

Submission for the **AMD Developer Hackathon** (LabLab.ai, May 2026) — **Track 3: Vision & Multimodal AI**.

## How it works

```
webcam frames  →  MediaPipe Holistic   →  trained sign classifier
   (1–5 fps)        (543-dim pose)        (WLASL Top-100 + alphabet)
                                                  │
                                                  ▼
                                      Llama-3.1-8B sentence composer
                                                  │
                                                  ▼
                                            Coqui XTTS-v2  →  speech
```

All four stages run **concurrently on a single AMD Instinct MI300X** via AMD Developer Cloud. Total weights ~22 GB on a 192 GB GPU — fits with margin for KV cache + serving overhead.

## V1 use cases

1. **ASL fingerspelling alphabet** — sign A–Z and 0–9 → AI speaks the letters / numbers
2. **Top-50 WLASL signs** (hello, thank you, name, please, sorry, family, eat, drink, work, …) → AI composes grammatical English sentences

V1 is **one-way**: deaf signs → hearing hears. Reverse direction (speech → on-screen text) is V2.

## Why AMD

The MI300X's 192 GB HBM3 and 5.3 TB/s memory bandwidth let the entire multi-stage pipeline (sign classifier + Llama-3.1-8B + XTTS-v2) run concurrently on a single GPU. Bandwidth-bound streaming workload is the textbook MI300X use case. Practical accessibility tools running globally need the cost-and-availability profile that AMD enables.

## Why this matters (business case)

Sign-language interpreters cost **$50–200 per hour** and are scarce. Courts, hospitals, schools, and public services **must by law** provide interpretation (ADA Title II/III in the US, EAA 2025 in the EU). Sorenson VRS — the dominant relay-services provider — books **$4B+ in annual revenue** in this space. SignBridge is the open-source backbone that any country, NGO, or enterprise can deploy on their own AMD compute.

## Privacy

Session-only. Frames and audio are processed in-memory and not persisted server-side beyond the WebSocket / HTTP session.

## Local dev

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN, AMD_DEV_CLOUD_*, OPENAI_API_KEY (fallback)

# Run the Gradio app
python app.py

# Run the inference backend (point at AMD Dev Cloud or local ROCm)
python -m signbridge.backend

# Train the classifier on WLASL Top-100 (Day 2 task — run on AMD Dev Cloud)
python -m signbridge.scripts.train_classifier --dataset data/wlasl --epochs 30
```

## Datasets used

- [WLASL](https://github.com/dxli94/WLASL) — Word-Level American Sign Language; we use the Top-100 subset
- ASL fingerspelling alphabet (open dataset)

## Models pulled from Hugging Face Hub

- `meta-llama/Llama-3.1-8B-Instruct` — sentence composer
- `coqui/XTTS-v2` — text-to-speech
- (V2 stretch) `openai/whisper-large-v3` — for the reverse direction

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Status

Active development — see `CLAUDE.md` for the working state and `docs/walkthrough.md` for the technical writeup.
