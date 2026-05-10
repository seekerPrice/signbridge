# SignBridge — paste-ready lablab.ai submission

> Submission deadline: **2026-05-11 03:00 Malaysia Time** (= Sunday May 10 12:00 PM Pacific Time).
> Open https://lablab.ai/ai-hackathons/amd-developer → bottom of page → **Submit Project**.
> Each block below maps 1:1 to a form field. Paste verbatim.

---

## Project Title (≤70 chars)

```
SignBridge — Real-time ASL → speech, fine-tuned Qwen3-VL on AMD MI300X
```

(70 characters; leads with the Track 2 fine-tune story.)

---

## Short Description (~150 chars)

```
Two people who couldn't communicate, now can. Real-time ASL → English speech, powered by Qwen3-VL we fine-tuned on AMD MI300X.
```

(132 characters.)

---

## Long Description (~350 words)

```
SignBridge is a real-time American Sign Language → English speech translator built for the AMD Developer Hackathon, Track 3 (Vision & Multimodal AI). We fine-tuned Qwen3-VL-8B on a single AMD Instinct MI300X and serve it natively through vLLM's video understanding API.

The user signs at the webcam — fingerspelled letters (Snapshot tab) or full motion words (Record sign tab) — and SignBridge replies in spoken English. Two people who couldn't communicate, now can.

Architecture: a hybrid pipeline. (1) MediaPipe Hand → trained MLP classifier handles static fingerspelling at 90% accuracy and 50ms latency on CPU — the textbook approach for static-pose tasks. (2) For motion words the recorded webcam clip is transcoded by ffmpeg and sent natively to a LoRA-fine-tuned Qwen3-VL-8B via vLLM's video_url block — Qwen3-VL processes the entire clip with its own temporal encoder rather than us pre-sampling frames. The fine-tune was 54 minutes on a single AMD Instinct MI300X and lifts ASL accuracy from 19% zero-shot to 92% in transformers eval. (3) Qwen3-8B composes the recognised sign tokens into natural English; gTTS turns the sentence into speech. Both LLMs run concurrently on the same MI300X via vLLM 0.17.1 on ROCm 7.2.

The MI300X did three jobs in this project on a single GPU: (1) ran the LoRA fine-tune in 54 minutes; (2) hosts the merged Qwen3-VL-8B for inference; (3) hosts the 8B composer in parallel. 192 GB HBM3 means we never had to reload weights or shard. The same workload on NVIDIA H100 (80 GB) would need a 3-GPU cluster.

Fine-tune artefacts (verifiable by judges): the merged Qwen3-VL-8B-ASL is public at huggingface.co/LucasLooTan/signbridge-qwen3vl-8b-asl. The MediaPipe-MLP classifier is at huggingface.co/LucasLooTan/signbridge-asl-classifier. Both pulled at runtime via hf_hub_download.

Why this matters: ASL interpreters cost $50–200 per hour and are scarce. Sorenson VRS books $4B+ in annual revenue filling this gap. SignBridge is an open-source MIT-licensed substrate that any Deaf-led NGO, school, ministry, or enterprise can deploy on their own AMD compute.

V1 is ASL-only by design — sign languages aren't interchangeable, and Deaf-led teams should own their own deployments. Built solo by Lucas Loo Tan Yu Heng, May 5–11, 2026.
```

---

## Technology & Category Tags

Pick from lablab dropdown:

**Primary (must select):**
- `Qwen` and/or `Qwen3-VL`
- `AMD Developer Cloud`
- `AMD ROCm`
- `HuggingFace Spaces`

**Secondary (relevant):**
- `LLaMA` (no — we replaced this with Qwen3-8B; skip)
- `Gradio`
- `FastAPI`
- `Vision`
- `Multimodal`
- `Accessibility`
- `Open Source`
- `vLLM`

**Track:** **Track 3 — Vision & Multimodal AI** (also satisfies Track 2 fine-tuning narrative if dual-track allowed)

---

## Pipeline at a glance (May 10 — current shipping)

Paste this block anywhere a one-screen architecture summary is needed (lablab form, slide notes, README):

```
- Static fingerspelling: MediaPipe Hand → trained MLP classifier (90% accuracy, ~50 ms on CPU)
- Motion signs: webcam recording → ffmpeg (480p, 8 fps, ≤4 s, H.264) → vLLM /v1/chat/completions
                 with a video_url block → fine-tuned Qwen3-VL-8B on AMD MI300X
- Sentence composer: Qwen3-8B on the same MI300X (vLLM, separate port)
- Speech synthesis: gTTS (Google's free TTS, fast, MP3 output)
- Live demo: HF Space (Gradio Docker SDK) — both tabs, end-to-end
```

---

## Cover Image

Upload `assets/cover.png` from the repo (1280×640 PNG, indigo→pink gradient with 🤟 + project name).

---

## Video Presentation

Paste the **YouTube Unlisted URL** of your demo video.

Reference shot list: `docs/demo-video-script.md`.

---

## Slide Presentation

Upload the **deck PDF**.

Build from `docs/pitch-deck.md`:
1. Open Google Slides → blank deck
2. Paste each slide's content into a blank slide
3. File → Download → PDF
4. Upload here

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

Before clicking Submit:

- [ ] Title pasted (70 chars)
- [ ] Short description pasted (132 chars)
- [ ] Long description pasted (~350 words)
- [ ] Tags selected (at minimum: Qwen, AMD Developer Cloud, AMD ROCm, HuggingFace Spaces)
- [ ] Cover image uploaded (`assets/cover.png`)
- [ ] Video URL pasted (YouTube unlisted)
- [ ] Pitch deck PDF uploaded
- [ ] GitHub URL pasted
- [ ] HF Space URL pasted
- [ ] **Track selection: Track 3 — Vision & Multimodal AI**
- [ ] Open Space in incognito → confirm it loads
- [ ] GitHub repo public + has clean README
- [ ] LICENSE file is MIT

When all boxes ticked → click Submit → wait for confirmation email → done.

**Aim to submit by 2026-05-11 02:00 MYT** (1-hour buffer before the 03:00 cutoff).
