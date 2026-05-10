# SignBridge — what only Lucas can do

> Status (2026-05-10): **submission deadline 03:00 MYT — ~5 hours left.** All written content + the .pptx deck are produced. Two things still need a human.

## 1 — Record the 2-min demo video

Follow `docs/demo-video-script.md`. Tools: QuickTime Player (Mac) for screen + camera capture, iMovie or CapCut for editing.

**Minimum viable shot list** (if pressed for time, do only these):

1. **Hook (10 s):** plain text card: *"70 million deaf people. Interpreters cost $50–200/hr. They're scarce."*
2. **Snapshot demo (30 s):** screen recording of `huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge`. Sign L-U-C-A-S letter-by-letter (📷 button per letter) → click 🔊 Speak → app says "Lucas."
3. **Record-sign demo (30 s):** switch to Record sign tab → record HELLO for ~1.5 s → click Submit → app says "hello (85%)" → click Speak → audio plays.
4. **Architecture flash (20 s):** show one slide from `assets/pitch-deck.pptx` — slide 4 (Architecture). Voiceover: *"Fine-tuned Qwen3-VL-8B handles motion ASL natively via vLLM video_url, Qwen3-8B composes English, gTTS speaks. All on a single AMD Instinct MI300X."*
5. **Close (10 s):** GitHub URL + HF Space URL + "🤟 SignBridge — MIT licensed."

**Hard rules:**
- Mention "AMD MI300X" by name ≥3 times in voice-over.
- Mention "Qwen3-VL" by name ≥2 times (Qwen Special Reward eligibility).
- Burn in subtitles for accessibility.
- Length: 2:00–2:30 max. Lablab cuts long videos.

**After recording:** upload to YouTube as **Unlisted**, copy the URL, paste into the lablab.ai form's "Video Presentation" field.

## 2 — Submit the lablab.ai form

Open https://lablab.ai/ai-hackathons/amd-developer → scroll to bottom → click **Submit Project**.

Use **`docs/SUBMIT_NOW.md`** for paste-ready content. Each block in that file maps 1:1 to a form field. The most important fields:

| Form field | Where to copy from |
|---|---|
| Project Title | `SUBMIT_NOW.md` first code block |
| Short Description | second code block (132 chars) |
| Long Description | third code block (~350 words, **already updated** with current pipeline) |
| Tags | tag list (Qwen, AMD Developer Cloud, AMD ROCm, HuggingFace Spaces, Vision, Multimodal, Accessibility, Open Source, Gradio, FastAPI, vLLM) |
| Cover Image | upload `assets/cover.png` (1280×640 PNG) |
| Video Presentation | YouTube unlisted URL from step 1 above |
| Slide Presentation | upload `assets/pitch-deck.pptx` (38.5 KB, 8 slides — already generated) |
| Public GitHub Repository | `https://github.com/seekerPrice/signbridge` |
| Demo Application Platform | `Hugging Face Space` |
| Application URL | `https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge` |
| **Track** | **Track 3 — Vision & Multimodal AI** |

**Pre-submit sanity check (do these 5 in incognito Chrome):**
- [ ] HF Space URL loads — Snapshot tab visible, camera placeholder visible.
- [ ] GitHub repo URL loads — README + LICENSE visible, license is MIT.
- [ ] HF Space Settings → Variables and secrets has `SIGNBRIDGE_VLM_MODEL=signbridge-qwen3vl-8b-asl` set (otherwise Record-sign returns 404).
- [ ] Video URL (YouTube) is publicly accessible — open in incognito to confirm.
- [ ] `assets/pitch-deck.pptx` opens in Google Slides / Keynote / PowerPoint without errors.

When all 5 ticked → Submit form → wait for confirmation email → done.

**Aim to submit by 02:00 MYT** (1-hour buffer before 03:00 cutoff).

---

## Done by Claude (you don't need to touch)

- [x] All `docs/` content updated to reflect current shipping pipeline (Qwen3-VL native video, gTTS, Qwen3-8B composer).
- [x] `signbridge/space.py` Record-sign tab description updated (no more "samples 4 frames").
- [x] `assets/pitch-deck.pptx` generated from `docs/pitch-deck.md` (8 slides, 16:9, 38.5 KB).
- [x] `assets/cover.png` is the existing 1280×640 indigo→pink gradient (verified, no regenerate needed).
- [x] `signbridge/scripts/build_pitch_deck.py` script for re-generating the deck if you want edits.
- [x] All commits pushed to HF Space + GitHub mirror.
