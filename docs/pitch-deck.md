# SignBridge — Pitch Deck (8 slides)

> Open a Google Slides deck (or Pitch). Paste each slide's content into the matching blank slide. Visuals are described in italics — replace with actual screenshots / diagrams / table renders.
> Aspect ratio: 16:9. Theme: indigo→pink gradient (matches HF Space card).

---

## Slide 1 — Title

**Title (huge):**
SignBridge

**Subtitle:**
Real-time ASL → English speech, on a single AMD Instinct MI300X.

**Footer (small):**
Track 3 · Vision & Multimodal AI · AMD Developer Hackathon 2026 · Lucas Loo Tan Yu Heng

*Visual: the cover.png we already shipped (1280×640 indigo→pink gradient with 🤟 + project name).*

---

## Slide 2 — The problem

**Headline:**
70 million deaf people. Sign-language interpreters cost $50–200 per hour. They're scarce.

**Body bullets:**
- Courts, hospitals, schools, public services **must by law** provide interpretation (ADA Title II/III in the US; European Accessibility Act 2025 in the EU).
- **Sorenson VRS**, the dominant sign-language relay-services provider, books **$4B+ in annual revenue** filling this gap — proof the demand is enormous and budgeted-for.
- Existing AI alternatives (Be My Eyes, Microsoft Seeing AI) are turn-based, photo-only, English-default, and closed-source. Real ASL is *motion* — they fundamentally can't translate "HELLO" or "THANK YOU".

*Visual: a row of three context icons — courthouse / hospital / classroom — labeled with the mandates.*

---

## Slide 3 — The solution

**Headline:**
Hold to record. Sign. Speak.

**Body (3-step arc):**
1. **Hold-to-record button** captures 1.5 seconds of your sign.
2. A multi-stage pipeline (vision → reasoning → speech) translates it.
3. The other person hears natural English.

**Tag line under the arc:**
Two people who couldn't communicate, now can.

*Visual: 3 screenshots of the live Gradio Space — (a) user signing into webcam; (b) "detected: HELLO (85%)"; (c) audio waveform playing "Hello.".*
*If single screenshot: just the Gradio "Record sign" tab mid-demo.*

---

## Slide 4 — Architecture (the AMD pitch)

**Headline:**
The whole pipeline fits on a single MI300X. NVIDIA H100 doesn't.

**Diagram (build in Slides; described as bullets):**
```
[ Webcam frame burst (4 frames, 1.5 s) ]
              │
              ▼
[ Qwen3-VL-8B  ── frame summariser, multi-image VLM call ]
              │
              ▼
[ Llama-3.1-8B ── sentence composer (sign tokens → English) ]
              │
              ▼
[ Coqui XTTS-v2 ── multilingual streaming TTS ]
              │
              ▼
[ Audio out ── speaker / Gradio audio component ]
```

**Comparison table (small print under diagram):**

| Component | Weights (FP16) | MI300X 1× (192 GB) | H100 80 GB |
|---|---|---|---|
| Qwen3-VL-8B | ~16 GB | ✅ fits | ✅ |
| Llama-3.1-8B | ~16 GB | ✅ fits | ✅ |
| XTTS-v2 + Whisper (V2) | ~5 GB | ✅ fits | ⚠ tight |
| (V2) **Llama-3.1-70B FP8 reasoner** | ~70 GB | **✅ still fits** | **❌ doesn't fit at all** |

**Closer:** The single-GPU concurrency story is the AMD pitch.

*Visual: the diagram + table as a single composite slide. Use a brand colour for the AMD column to highlight.*

---

## Slide 5 — Live demo

**Headline:**
*(blank — this slide is the live demo)*

**Speaker note:**
Switch to the live HF Space at huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge. 30 seconds:
1. **Snapshot tab** — fingerspell L-U-C-A-S → click Speak → AI says "Lucas."
2. **Record sign tab** — record HELLO → click Submit → "hello" detected → click Speak → AI says "Hello."

If demo fails / network down → fall back to the pre-recorded 2-min video on slide 6.

*Visual: leave the slide blank or use a single QR code linking to the Space URL for the audience to scan and try themselves.*

---

## Slide 6 — Demo video (fallback)

**Headline:**
*(blank — this slide embeds the demo video)*

**Embed:**
The 2–3 minute demo video, looping, autoplay-on-slide-show.

*Visual: video player.*

---

## Slide 7 — Why this is the right submission for Track 3

**Headline:**
Four judging criteria, four deliberate choices.

**Two-column layout:**

| Judging criterion | Our choice |
|---|---|
| **Application of Technology** | Multi-modal pipeline (vision + reasoning + voice) running concurrently on a single MI300X — exactly what Track 3's "massive memory bandwidth of AMD GPUs" was for. |
| **Presentation** | Demo is *experienced*: judge holds phone, signs HELLO, hears "Hello." 30 seconds, no explanation needed. |
| **Business Value** | $4B+ existing market (Sorenson VRS comparable), legally-mandated interpretation budgets, open-source so any Deaf-led NGO / ministry / school can self-host on their own AMD compute. |
| **Originality** | Streaming continuous multi-frame VLM agent for sign language — no peer-reviewed benchmark exists for this approach yet (we checked the literature). Real ASL motion-words, not just fingerspelling. |

*Visual: 2×2 grid of icons, one per criterion.*

---

## Slide 8 — Substrate, not product · Open · Deaf-led future

**Headline:**
SignBridge is a substrate. Deaf-led teams are the deployers.

**Body:**
- **MIT-licensed**, code at github.com/seekerPrice/signbridge — anyone can self-host.
- **ASL only V1 is a scope decision.** BSL, MSL, CSL, ISL, +200 sign languages each deserve their own teams, training data, and Deaf community leadership. (Citing Bragg et al., *"Systemic Biases in Sign Language AI Research"*, arXiv 2403.02563.)
- **Privacy by default** — frames and audio are processed in-memory and not persisted server-side beyond the request lifetime.

**Closing line (large):**
The hardest part of accessibility isn't building. It's deploying. AMD makes the deploying possible.

*Visual: world map outline with sign-language regional dots; or just the SignBridge logo with the closing tagline.*

---

## Speaker-note tips (read these before recording)

1. **Lead with the human problem (Slide 2), not the architecture.** Architecture is for criterion 1; emotion is what closes criteria 2–4.
2. **Time the live demo** — 30 seconds max. If it fails, switch to fallback video without comment.
3. **Always say "AMD MI300X" by name** at least 3 times in the talk track. Sponsors notice.
4. **End on the substrate framing** — pre-empts the "savior tech" critique that Deaf-AI judges look out for.

---

## Export

Once filled in: File → Download → PDF document → upload to lablab.ai submission form's "Slide Presentation" field.
