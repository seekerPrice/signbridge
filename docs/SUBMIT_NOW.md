# SignBridge — paste-ready lablab.ai submission

> Submission deadline: **2026-05-11 03:00 Malaysia Time** (= Sunday May 10 12:00 PM Pacific Time).
> Open https://lablab.ai/ai-hackathons/amd-developer → bottom of page → **Submit Project**.
> Each block below maps 1:1 to a form field. Paste verbatim.

---

## Project Title (form max: 50 chars, min 5)

```
SignBridge — fine-tuned Qwen3-VL on AMD MI300X
```

(47 characters; leads with Qwen + AMD for both the Qwen Special Reward and Track 3 narratives.)

---

## Short Description (form max: 255 chars, min 50)

```
Two people who couldn't communicate, now can. Real-time ASL → English speech, powered by Qwen3-VL we fine-tuned on AMD MI300X.
```

(126 characters — fits comfortably.)

---

## Long Description (form max: 2000 chars, min 600)

```
SignBridge is a real-time American Sign Language → English speech translator built for the AMD Developer Hackathon, Track 3 (Vision & Multimodal AI). We fine-tuned Qwen3-VL-8B on a single AMD Instinct MI300X and serve it natively through vLLM's video understanding API.

The user signs at the webcam — fingerspelled letters (Snapshot tab) or full motion words (Record sign tab) — and SignBridge replies in spoken English. Two people who couldn't communicate, now can.

Architecture: (1) MediaPipe Hand → trained MLP classifier handles static fingerspelling at 90% accuracy, ~50 ms on CPU. (2) For motion words the webcam clip is transcoded with ffmpeg and sent natively to a LoRA-fine-tuned Qwen3-VL-8B via vLLM's video_url block — Qwen3-VL processes the clip with its own temporal encoder, no manual frame sampling. The 54-minute LoRA on a single MI300X lifts ASL accuracy from 19% zero-shot to 92% in transformers eval. (3) Qwen3-8B composes recognised tokens into English; gTTS speaks it. Both LLMs run concurrently on the same MI300X via vLLM 0.17.1 on ROCm 7.2.

One MI300X did three jobs on one GPU: ran the LoRA fine-tune (54 min), hosts the merged Qwen3-VL-8B for inference, and hosts the 8B composer in parallel. 192 GB HBM3 means no swapping or sharding. The same workload on H100 (80 GB) needs a 3-GPU cluster.

Fine-tune artefacts (judge-verifiable): merged Qwen3-VL-8B-ASL at huggingface.co/LucasLooTan/signbridge-qwen3vl-8b-asl; MediaPipe-MLP classifier at huggingface.co/LucasLooTan/signbridge-asl-classifier. Both pulled at runtime via hf_hub_download.

Why it matters: ASL interpreters cost $50–200/hr and are scarce. Sorenson VRS books $4B+/yr filling this gap. SignBridge is MIT-licensed open source — any Deaf-led NGO, school, ministry can self-host on their own AMD compute. V1 is ASL-only by design; sign languages aren't interchangeable.

Built solo by Lucas Loo Tan Yu Heng, May 5–11, 2026.
```

(~1980 chars — fits the 2000 max with ~20 char buffer.)

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
