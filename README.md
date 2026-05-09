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
                  ┌─► MediaPipe Hand → trained MLP    (90% acc, 50ms CPU)
webcam frame ────┤                       │
                  └─► fine-tuned Qwen3-VL-8B (LoRA on AMD MI300X)
                                          │      (92% acc, motion + fallback)
                                          ▼
                          Qwen3-8B sentence composer
                                          │   (AMD MI300X)
                                          ▼
                              Coqui XTTS-v2 TTS
                                          │
                                          ▼
                                       🔊 speech
```

A hybrid pipeline: a small classical-ML classifier handles static fingerspelling at 90% accuracy with 50 ms CPU latency; a LoRA-fine-tuned Qwen3-VL-8B handles motion-dependent signs and ambiguous static frames; Qwen3-8B turns sign tokens into natural English. The two LLMs run **concurrently on a single AMD Instinct MI300X** via vLLM 0.17.1 on ROCm 7.2 — combined ~34 GB on a 192 GB GPU.

The fine-tune itself was trained on a single MI300X in **54 minutes** with LoRA (rank 16, target q/k/v/o, 2 epochs on 9,786 ASL Alphabet samples). Final eval loss 0.48; gold-set accuracy 92.3% — a 4.8× lift over the 19.2% zero-shot baseline.

- Fine-tuned model: `huggingface.co/LucasLooTan/signbridge-qwen3vl-8b-asl`
- Landmark classifier: `huggingface.co/LucasLooTan/signbridge-asl-classifier`

## V1 use cases

1. **ASL fingerspelling alphabet** — sign A–Z and 0–9 → AI speaks the letters / numbers
2. **Top-50 WLASL signs** (hello, thank you, name, please, sorry, family, eat, drink, work, …) → AI composes grammatical English sentences

V1 is **one-way**: deaf signs → hearing hears. Reverse direction (speech → on-screen text) is V2.

## Why AMD

The MI300X did three jobs in this project on a single GPU: (1) ran the LoRA fine-tune of Qwen3-VL-8B in 54 minutes; (2) hosts the merged model for inference via vLLM; (3) hosts the Qwen3-8B composer in parallel for sentence composition. 192 GB HBM3 means we never had to reload weights, swap, or shard between training and serving. NVIDIA H100 (80 GB) would require a 3-GPU cluster for the same V2 70B reasoner upgrade — practical accessibility tools running globally need the cost-and-availability profile that AMD enables.

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
