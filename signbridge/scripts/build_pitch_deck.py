"""Build the lablab.ai pitch deck as a .pptx file.

Output: assets/pitch-deck.pptx — 8 slides matching docs/pitch-deck.md.
Usage:  .venv/bin/python -m signbridge.scripts.build_pitch_deck

User can then upload to Google Slides (File → Open → Upload) or
directly to the lablab.ai submission form's "Slide Presentation" field.

This is a one-shot generator — it doesn't try to be a templating engine.
Each slide is hand-written below so we can position elements precisely.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu, Inches, Pt

# 16:9 layout
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Brand palette (matches the indigo→pink HF Space theme)
INDIGO = RGBColor(0x4F, 0x46, 0xE5)
INDIGO_DARK = RGBColor(0x1E, 0x1B, 0x4B)
PINK = RGBColor(0xEC, 0x48, 0x99)
SLATE = RGBColor(0x47, 0x55, 0x69)
SLATE_LIGHT = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NEAR_WHITE = RGBColor(0xF8, 0xFA, 0xFC)


def _add_text(slide, x, y, w, h, text, *, size=18, bold=False, color=INDIGO_DARK, align=None):
    """Add a text box; returns the text frame for further tweaking."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tf


def _add_bullets(slide, x, y, w, h, lines, *, size=16, color=INDIGO_DARK):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        run = p.add_run()
        run.text = f"• {line}"
        run.font.size = Pt(size)
        run.font.color.rgb = color


def _add_band(slide, *, color=INDIGO, height_inches=0.6):
    """Decorative bottom band."""
    band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        0, SLIDE_H - Inches(height_inches),
        SLIDE_W, Inches(height_inches),
    )
    band.fill.solid()
    band.fill.fore_color.rgb = color
    band.line.fill.background()
    band.shadow.inherit = False


def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Big title
    _add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(2),
              "🤟 SignBridge", size=88, bold=True, color=INDIGO)
    _add_text(s, Inches(0.7), Inches(3.5), Inches(12), Inches(1.5),
              "Real-time ASL → English speech, on a single AMD Instinct MI300X.",
              size=28, color=SLATE)
    _add_text(s, Inches(0.7), Inches(6.4), Inches(12), Inches(0.6),
              "Track 3 · Vision & Multimodal AI · AMD Developer Hackathon 2026 · Lucas Loo Tan Yu Heng",
              size=14, color=SLATE)
    _add_band(s, color=INDIGO)
    return s


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(0.5), Inches(12), Inches(1.2),
              "70 million deaf people. Interpreters cost $50–200/hr. They're scarce.",
              size=32, bold=True, color=INDIGO_DARK)
    _add_bullets(s, Inches(0.7), Inches(2.2), Inches(12), Inches(4.5), [
        "Courts, hospitals, schools, and public services must by law provide interpretation (ADA Title II/III in the US; European Accessibility Act 2025 in the EU).",
        "Sorenson VRS — the dominant sign-language relay-services provider — books $4B+ in annual revenue filling this gap. The demand is enormous and budgeted-for.",
        "Existing AI alternatives (Be My Eyes, Microsoft Seeing AI) are turn-based, photo-only, English-default, and closed-source.",
        "Real ASL is motion. Single-frame approaches fundamentally cannot translate \"HELLO\" or \"THANK YOU\".",
    ], size=18)
    _add_band(s, color=PINK)
    return s


def slide_solution(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(0.5), Inches(12), Inches(1.2),
              "Hold to record. Sign. Speak.", size=40, bold=True, color=INDIGO)
    _add_bullets(s, Inches(0.7), Inches(2.0), Inches(12), Inches(4), [
        "1. Hold-to-record button captures 1.5 seconds of your sign.",
        "2. Multi-stage pipeline (vision → reasoning → speech) translates it.",
        "3. The other person hears natural English.",
    ], size=22)
    _add_text(s, Inches(0.7), Inches(5.5), Inches(12), Inches(1.5),
              "Two people who couldn't communicate, now can.",
              size=28, bold=True, color=PINK)
    _add_band(s, color=INDIGO)
    return s


def slide_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(0.4), Inches(12), Inches(1),
              "We fine-tuned Qwen3-VL-8B on a single MI300X — 54 minutes, 92% accuracy.",
              size=26, bold=True, color=INDIGO)
    diagram = (
        "Webcam frame\n"
        "  ├─►  MediaPipe Hand → trained MLP   (90% acc, ~50 ms CPU)\n"
        "  │      └─ falls through to ↓\n"
        "  └─►  Recorded clip → ffmpeg → vLLM video_url\n"
        "         → fine-tuned Qwen3-VL-8B (native video, AMD MI300X)\n"
        "                  ↓\n"
        "         Qwen3-8B composer  (sign tokens → English, vLLM port 8001)\n"
        "                  ↓\n"
        "         gTTS  (free, fast speech synthesis)\n"
        "                  ↓\n"
        "             Audio out"
    )
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.6), Inches(8), Inches(4.5))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = diagram
    run.font.size = Pt(14)
    run.font.name = "Menlo"
    run.font.color.rgb = INDIGO_DARK

    _add_bullets(s, Inches(8.9), Inches(1.6), Inches(4.1), Inches(5.0), [
        "MI300X 1× holds the entire pipeline.",
        "Same workload on H100 (80 GB) → 3-GPU cluster.",
        "192 GB HBM3, 5.3 TB/s mem bandwidth.",
        "Both LLMs concurrent on one GPU, no sharding.",
    ], size=14, color=SLATE)
    _add_band(s, color=PINK)
    return s


def slide_demo(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(2.5), Inches(12), Inches(2),
              "Live demo.", size=72, bold=True, color=INDIGO,
              align=None)
    _add_text(s, Inches(0.7), Inches(4.0), Inches(12), Inches(1.5),
              "huggingface.co/spaces/lablab-ai-amd-developer-hackathon/signbridge",
              size=20, color=SLATE)
    _add_text(s, Inches(0.7), Inches(5.5), Inches(12), Inches(1),
              "(Switch to the live HF Space — fingerspell L-U-C-A-S → Speak → \"Lucas\")",
              size=14, color=SLATE)
    _add_band(s, color=INDIGO)
    return s


def slide_qwen_focus(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(0.4), Inches(12), Inches(1.2),
              "LoRA-fine-tuned Qwen3-VL-8B — the brain.",
              size=32, bold=True, color=INDIGO)
    _add_bullets(s, Inches(0.7), Inches(1.8), Inches(12), Inches(5), [
        "Recognizer: our LoRA-fine-tuned Qwen3-VL-8B (huggingface.co/LucasLooTan/signbridge-qwen3vl-8b-asl), trained in 54 min on a single AMD Instinct MI300X. Lifts ASL accuracy from 19% zero-shot → 92%.",
        "Motion signs: we send the whole recorded clip natively to Qwen3-VL via vLLM's video_url block. Qwen3-VL's own temporal encoder handles motion. No manual frame sampling.",
        "Closed-vocabulary forcing + domain priming keep Qwen on-rails for the 87-token sign vocab.",
        "Qwen3-8B composes Qwen-VL's tokens into English (also on the MI300X via vLLM, separate port). gTTS synthesises the audio.",
    ], size=16, color=INDIGO_DARK)
    _add_text(s, Inches(0.7), Inches(6.3), Inches(12), Inches(0.7),
              "Qwen3-VL is the only thing in the pipeline making the visual judgement. The rest is plumbing.",
              size=14, bold=True, color=PINK)
    _add_band(s, color=INDIGO)
    return s


def slide_judging(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(0.4), Inches(12), Inches(1),
              "Four judging criteria. Four deliberate choices.",
              size=28, bold=True, color=INDIGO)
    rows = [
        ("Application of Technology",
         "Multi-modal pipeline (vision + reasoning + voice) running concurrently on a single MI300X — exactly what Track 3's massive memory bandwidth was for."),
        ("Presentation",
         "Demo is experienced: judge holds phone, signs HELLO, hears \"Hello.\" 30 seconds, no explanation needed."),
        ("Business Value",
         "$4B+ existing market (Sorenson VRS comparable), legally-mandated interpretation budgets, open-source so any Deaf-led NGO/ministry/school can self-host on their own AMD compute."),
        ("Originality",
         "First open-source pipeline to send recorded ASL natively to a fine-tuned Qwen3-VL via vLLM video_url — combining the AMD fine-tune story with native-video understanding for sign language."),
    ]
    y = Inches(1.6)
    for header, body in rows:
        _add_text(s, Inches(0.7), y, Inches(3.5), Inches(1.0),
                  header, size=16, bold=True, color=PINK)
        _add_text(s, Inches(4.4), y, Inches(8.5), Inches(1.3),
                  body, size=14, color=INDIGO_DARK)
        y += Inches(1.35)
    _add_band(s, color=PINK)
    return s


def slide_substrate_close(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(s, Inches(0.7), Inches(0.5), Inches(12), Inches(1.2),
              "SignBridge is a substrate. Deaf-led teams are the deployers.",
              size=28, bold=True, color=INDIGO)
    _add_bullets(s, Inches(0.7), Inches(2.0), Inches(12), Inches(3.0), [
        "MIT-licensed, open-source: github.com/seekerPrice/signbridge",
        "ASL only V1 is a scope decision — BSL, MSL, CSL, ISL, +200 sign languages each deserve their own teams, training data, and Deaf community leadership.",
        "Privacy by default — frames and audio are processed in-memory, not persisted.",
    ], size=18, color=INDIGO_DARK)
    _add_text(s, Inches(0.7), Inches(5.4), Inches(12), Inches(1.5),
              "The hardest part of accessibility isn't building. It's deploying.",
              size=22, bold=True, color=SLATE)
    _add_text(s, Inches(0.7), Inches(6.3), Inches(12), Inches(0.8),
              "AMD makes the deploying possible.",
              size=22, bold=True, color=PINK)
    _add_band(s, color=INDIGO)
    return s


def main() -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_title(prs)
    slide_problem(prs)
    slide_solution(prs)
    slide_architecture(prs)
    slide_demo(prs)
    slide_qwen_focus(prs)
    slide_judging(prs)
    slide_substrate_close(prs)

    out = Path(__file__).parents[2] / "assets" / "pitch-deck.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
