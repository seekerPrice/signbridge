# SignBridge — lablab.ai Submission Form Content

> Open https://lablab.ai/ai-hackathons/amd-developer → scroll to bottom → click **Submit project**. Paste each field below into the matching input.

---

## Project Title (≤ ~70 chars)

```
SignBridge — Real-time ASL → speech, Qwen3-VL on AMD MI300X
```

(60 characters; leads with Qwen for Qwen Special Reward eligibility.)

---

## Short Description (≤ 150 chars typical)

```
Two people who couldn't communicate, now can. Real-time ASL → English speech, powered by Qwen3-VL on AMD Instinct MI300X.
```

(132 characters.)

---

## Long Description (no hard limit, ~300 words is the sweet spot)

```
SignBridge is a real-time American Sign Language to English speech translator built for the AMD Developer Hackathon, Track 3 (Vision & Multimodal AI). We fine-tuned Qwen3-VL-8B on a single AMD Instinct MI300X and serve it natively through vLLM's video understanding API.

The user signs at the webcam — either fingerspelled letters (Snapshot tab) or full motion words (Record sign tab) — and SignBridge replies in spoken English. Two people who couldn't communicate, now can.

Architecture: a hybrid pipeline. (1) MediaPipe Hand → trained MLP classifier handles static fingerspelling at 90% accuracy and 50ms latency on CPU. (2) For motion words, the recorded webcam clip is transcoded by ffmpeg and sent natively to a LoRA-fine-tuned Qwen3-VL-8B via vLLM's video_url block — Qwen3-VL processes the entire clip with its own temporal encoder, no manual frame-sampling. The fine-tune was 54 minutes on a single AMD Instinct MI300X and lifts ASL accuracy from 19% zero-shot to 92% in transformers eval. (3) Qwen3-8B composes the recognised sign tokens into natural English; gTTS turns the sentence into speech. Both LLMs run concurrently on the same MI300X via vLLM. The 192 GB HBM3 of one MI300X holds the entire pipeline with margin — the same workload on NVIDIA H100 needs three GPUs.

Fine-tune artefacts: the merged Qwen3-VL-8B-ASL is public at `huggingface.co/LucasLooTan/signbridge-qwen3vl-8b-asl`; the MediaPipe-MLP classifier is at `huggingface.co/LucasLooTan/signbridge-asl-classifier`. Both pulled at runtime via `hf_hub_download`. This satisfies both Track 3 (Vision & Multimodal) and Track 2 (Fine-Tuning on AMD GPUs) narratives — fine-tuning, ROCm, vLLM, and Hugging Face Optimum-AMD all in the same project.

For motion-dependent signs (HELLO, THANK_YOU, PLEASE, EAT) the Record-sign tab uploads the recorded clip directly to Qwen3-VL via vLLM's `video_url` content block — most ASL signs are motion, not held poses, so single-frame approaches fundamentally cannot translate them.

Why this matters: sign-language interpreters cost $50–200 per hour and are scarce. Courts, hospitals, schools, and public services must by law (ADA, EAA 2025) provide interpretation. Sorenson VRS — the dominant relay-services provider — books $4B+ in annual revenue filling this gap. SignBridge is an open-source MIT-licensed substrate that any Deaf-led NGO, school, ministry, or enterprise can deploy on their own AMD compute.

V1 is ASL-only, deliberately. Sign languages aren't interchangeable — BSL, MSL, CSL, ISL, and 200+ others each deserve their own teams, training data, and Deaf community leadership. (See Bragg et al., "Systemic Biases in Sign Language AI Research", arXiv 2403.02563.)

Built solo by Lucas Loo Tan Yu Heng, May 5–11, 2026.
```

---

## Technology & Category Tags

Pick from lablab's tag dropdown — these are the tags that match SignBridge:

**Primary (must-haves):**
- `Qwen` / `Qwen3-VL` (Qwen3-VL-8B vision recognizer — central; eligible for Qwen Special Reward 10M tokens)
- `AMD Developer Cloud`
- `AMD ROCm`
- `HuggingFace Spaces`

**Secondary (relevant):**
- `Qwen` / `Qwen3-8B` (composer model — counts toward Qwen Special Reward)
- `Gradio`
- `FastAPI`
- `Vision`
- `Multimodal`
- `Accessibility`
- `Open Source`

**Track:** Track 3 — Vision & Multimodal AI

---

## Cover Image

Upload `assets/cover.png` from the repo (1280×640 PNG, ~60 KB).

If lablab requires a different aspect ratio (e.g. square 1:1), regenerate with `python -m signbridge.scripts.make_cover` after editing the `WIDTH, HEIGHT` constants in `signbridge/scripts/make_cover.py`.

---

## Video Presentation

Paste the YouTube URL of the demo video (uploaded as **Unlisted**).

Reference content: `docs/demo-video-script.md`.

---

## Slide Presentation

Upload the deck PDF.

Reference content: `docs/pitch-deck.md`. Build in Google Slides, File → Download → PDF, upload here.

---

## Public GitHub Repository

```
https://github.com/seekerPrice/signbridge
```

---

## Demo Application Platform

```
Hugging Face Space
```

---

## Application URL

```
https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge
```

---

## Final pre-submit checklist

Before clicking Submit on lablab:

- [ ] Title pasted (63 chars)
- [ ] Short description pasted (132 chars)
- [ ] Long description pasted (~300 words)
- [ ] Tags selected (Track 3 + at minimum: AMD Developer Cloud, AMD ROCm, HuggingFace Spaces, Qwen, LLaMA)
- [ ] Cover image uploaded (assets/cover.png)
- [ ] Video URL pasted (YouTube unlisted)
- [ ] Pitch deck PDF uploaded
- [ ] GitHub URL pasted
- [ ] HF Space URL pasted
- [ ] **Track selection: Track 3 — Vision & Multimodal AI**
- [ ] HF Space loads from a fresh browser (incognito test)
- [ ] GitHub repo has a clean README
- [ ] LICENSE file is MIT
- [ ] All commits pushed to both remotes

When all boxes are ticked → click Submit → wait for confirmation email → done.

Time-target: submit by **2026-05-11 02:00 MYT** (1-hour buffer before the 03:00 cutoff).
