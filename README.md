---
title: SignBridge
emoji: 🤟
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
thumbnail: assets/cover.png
license: mit
short_description: Real-time ASL → English speech on AMD MI300X.
---

# SignBridge — real-time ASL → speech

Two people who couldn't communicate, now can.

A deaf person signs into the webcam. SignBridge — a multi-stage vision + reasoning + voice pipeline running on a single AMD Instinct MI300X — translates the signs into spoken English in under 2 seconds.

Submission for the **AMD Developer Hackathon** (LabLab.ai, May 2026) — **Track 3: Vision & Multimodal AI**.

## How it works

```
webcam frames  →  Qwen3-VL-32B  →  Qwen3-8B  →  Coqui XTTS-v2  →  speech
                  (sign vision)   (composer)   (TTS)
```

All three stages run **concurrently on a single AMD Instinct MI300X** via AMD Developer Cloud. Total weights ~34 GB (Qwen3-VL-32B + Qwen3-8B + XTTS-v2) on a 192 GB GPU — fits with margin for KV cache + serving overhead. Both LLMs are Qwen-family, served via vLLM 0.17.1 on ROCm 7.2.

## V1 use cases

1. **ASL fingerspelling alphabet** — sign A–Z and 0–9 → AI speaks the letters / numbers
2. **Top-50 WLASL signs** (hello, thank you, name, please, sorry, family, eat, drink, work, …) → AI composes grammatical English sentences

V1 is **one-way**: deaf signs → hearing hears. Reverse direction (speech → on-screen text) is V2.

## Why AMD

The MI300X's 192 GB HBM3 fits the entire pipeline (Qwen3-VL-32B + Llama-3.1-8B + XTTS-v2) on one GPU with margin. NVIDIA H100 (80 GB) requires sharding, and the V2 plan to upgrade to a 70B reasoner is impossible on H100 without a 3-GPU cluster. Single-GPU concurrency + 5.3 TB/s memory bandwidth is the actual AMD pitch — practical accessibility tools running globally need the cost-and-availability profile that AMD enables.

## Why this matters (business case)

Sign-language interpreters cost **$50–200 per hour** and are scarce. Courts, hospitals, schools, and public services **must by law** provide interpretation (ADA Title II/III in the US, EAA 2025 in the EU). Sorenson VRS — the dominant relay-services provider — books **$4B+ in annual revenue** in this space. SignBridge is the open-source backbone that any country, NGO, or enterprise can deploy on their own AMD compute.

## Privacy

Session-only. Frames and audio are processed in-memory and not persisted server-side beyond the WebSocket / HTTP session.

## For Deaf-led teams

SignBridge is open-source under MIT license and intentionally scoped to ASL-only V1. The pipeline is a substrate, not a finished product — Deaf-led organisations (schools-for-the-Deaf, NGOs, ministries) are the intended deployers. Other sign languages (BSL, MSL, CSL, ISL, +200 more) deserve their own teams, training data, and Deaf community leadership. See [`docs/walkthrough.md`](docs/walkthrough.md) → "Deployment ethics" for the design principles drawn from the Deaf-led academic literature.

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

- `Qwen/Qwen3-VL-32B-Instruct` — sign vision (recognizer)
- `Qwen/Qwen3-8B` — sentence composer
- `coqui/XTTS-v2` — text-to-speech
- (V2 stretch) `openai/whisper-large-v3` — for the reverse direction

## License

MIT. See [`LICENSE`](LICENSE).

## Status

Active development — see `CLAUDE.md` for the working state and `docs/walkthrough.md` for the technical writeup.
