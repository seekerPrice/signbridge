# SignBridge — Demo Video Script

> Target length: **2:30 (≤ 3 min)**. Format: 1080p MP4, MP3 audio. Aspect ratio 16:9.
> Tools: QuickTime Player (Mac) for screen + camera capture, iMovie or CapCut for editing.

---

## Story arc (3 acts)

| Time | Act | Beat |
|---|---|---|
| 0:00–0:20 | **Hook** | Open with the human problem; viewer must feel the gap. |
| 0:20–1:30 | **Demo** | Live SignBridge in action — both fingerspelling AND a motion sign. |
| 1:30–2:30 | **Why AMD + close** | Architecture diagram + concrete MI300X comparison + open-source ethics + URL. |

Hard rule: **no slide-by-slide voice-over reading**. The demo should *play live*; voice-over should narrate what we're seeing, not summarise text on screen.

---

## Shot list

### Act 1 — Hook (0:00 → 0:20)

**Visual A (5 s):** Plain background, bold text card fades in:
> 70 million deaf people. Interpreters cost $50–200 / hour. They're scarce.

**Visual B (5 s):** Text card → "What if your phone could just translate?"

**Visual C (10 s):** Camera shot of you (Lucas) in a quiet room, signing HELLO at the camera silently. No voice-over yet. Hold the silence — let the viewer feel that the sign means nothing to them.

**Voice-over:** *(starts at 0:15)*
> "Most of us can't read this. SignBridge can."

---

### Act 2 — Live demo (0:20 → 1:30)

**Setup (0:20 → 0:25):** 5-second screen-recording of the live HF Space loading at `huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge`. URL bar visible. Tabs visible: "Snapshot" and "Record sign". This proves it's a live deployed product, not a slide deck.

**Beat 2A — Fingerspelling (0:25 → 0:55):**

**Visual (split screen recommended):** Left = your face/hand on webcam, right = the Gradio app receiving frames.
- Sign **L** clearly. Click **Capture sign**. App shows "detected: L (85%)".
- Sign **U**. Capture.
- Sign **C**. Capture.
- Sign **A**. Capture.
- Sign **S**. Capture.
- Click **🔊 Speak**. App composes → speaks: **"Lucas."**

**Voice-over during this beat:**
> "First, fingerspelling. I sign each letter, the app captures it, and—" *(pause for the speak)* — *"composed in natural English."*

**Beat 2B — Motion sign (0:55 → 1:25):**

**Visual:** Switch tabs to **Record sign**. Hit Record, sign **HELLO** (the wave-from-forehead motion), stop, click Submit.
- Detected: **hello (85%)**. Click Speak.
- App says: **"Hello."**

Repeat one more sign for variety: **THANK_YOU**.

**Voice-over:**
> "But fingerspelling alone isn't real ASL — most signs are *motion*. Hold-to-record captures the whole gesture, not just one frame. The system detects the motion across frames and..." *(pause for the speak)*

**Beat 2C — Two-person scene (1:25 → 1:30):** *(optional but high-impact)*

**Visual:** You sign something to a hearing person; they hear the AI say it; they react. Hold the human reaction for 2 seconds.

**No voice-over** during this beat — let the moment land.

---

### Act 3 — Architecture + AMD pitch (1:30 → 2:30)

**Beat 3A — Architecture diagram (1:30 → 1:55):**

**Visual:** Static slide showing the pipeline:
```
Webcam frames → Qwen3-VL-8B (vision) → Llama-3.1-8B (composer) → XTTS-v2 (speech)
                            All on a single AMD Instinct MI300X
```

**Voice-over:**
> "Under the hood: Qwen3-VL-8B reads each frame, Llama-3.1 composes the sentence, XTTS speaks it — all running concurrently on a single AMD Instinct MI300X. Vision, reasoning, and voice on one GPU."

**Beat 3B — The MI300X comparison (1:55 → 2:15):**

**Visual:** The comparison table from the walkthrough:

| | MI300X 1× | H100 80 GB |
|---|---|---|
| V1 pipeline (~34 GB) | ✅ comfortable | ⚠ tight |
| V2 with Llama-3.1-70B FP8 (~70 GB extra) | ✅ still fits | ❌ doesn't fit |

**Voice-over:**
> "192 GB of HBM3. Same workload on NVIDIA H100 needs three GPUs. Practical accessibility tools running globally need the cost-and-availability profile that AMD enables."

**Beat 3C — Substrate + close (2:15 → 2:30):**

**Visual:** Final slide:
- "Open source, MIT — github.com/seekerPrice/signbridge"
- "Hugging Face Space — huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge"
- "ASL V1. Deaf-led teams own the rest."
- 🤟 SignBridge

**Voice-over:**
> "SignBridge is open source under MIT. It's a substrate — Deaf-led organisations deploy it for their own languages. The hardest part of accessibility isn't building. It's deploying. AMD makes the deploying possible. Thanks for watching."

---

## Voice-over recording tips

- Record voice **separately** from screen capture (better audio quality). Use QuickTime "New Audio Recording" with a mic 6–12 inches away.
- One take, then cut. Don't try to dub multiple takes line-by-line.
- Cadence: ~140 words/min. Pause for 0.5 s after each section.
- If you have a good pop filter / lavalier, use it. AirPods Pro built-in mic is workable but compresses dynamics.

---

## Editing notes

- **Captions/subtitles required.** Burn in the spoken English text below the speaker's face throughout — both for accessibility and so judges can follow with sound off.
- **Highlight the recognized token visually.** When the app shows "detected: hello (85%)", zoom in or add a brief highlight box on that text — judges' eyes need to find it fast.
- **Music: skip.** The demo is loud enough on its own; background music distracts from the speech-output beats.
- **Smooth transitions only** — don't use fancy wipes; cut on action.
- **Final cut export:** 1080p, H.264, MP4, ≤100 MB if possible (lablab uploader has size limits).

---

## Prep before recording

- [ ] AMD Dev Cloud credit landed (so the live demo uses MI300X — *this is the hackathon talk-track*); fall back to HF Inference if not.
- [ ] Lighting: front-facing soft light. No back-window glare.
- [ ] Plain background (white wall ideal).
- [ ] Wear a contrasting solid colour (not patterns) — VLM accuracy improves.
- [ ] Webcam height: at eye level. Hands need to be in frame for signs.
- [ ] Test the live HF Space URL once before recording. If it errors, fix before pressing record.
- [ ] One dry run end-to-end with a stopwatch. Trim if over 2:45.

---

## Recording order (don't shoot in story order)

1. **Live demo screen recording first** — 3 takes of the full demo flow, pick the cleanest.
2. **Voice-over second** — record continuous narration over the picked demo take.
3. **B-roll of you signing alone** (Act 1 silent shot, Act 2C two-person reaction) — last, since they're easier to re-shoot.
4. Edit it together in iMovie / CapCut.
5. Export.
6. Upload to YouTube as **Unlisted**, copy URL.
7. Paste URL into lablab.ai submission form's "Video Presentation" field.

---

## Export checklist

- [ ] Length 2:00–3:00
- [ ] Captions visible throughout
- [ ] AMD Dev Cloud / MI300X mentioned by name ≥3 times
- [ ] Qwen3-VL mentioned by name ≥2 times (Qwen Special Reward eligibility)
- [ ] HF Space URL shown on screen at least once
- [ ] GitHub URL shown on screen at least once
- [ ] No copyrighted music / footage
- [ ] Speaker face visible (judges remember faces)
- [ ] Final shot: SignBridge logo + URLs
