"""Gradio UI — the HF Space's user surface.

V1 UX is snapshot-driven (more reliable than continuous-streaming for an
agentic VLM-backed recognizer):
  1. Live webcam preview shows what's framed.
  2. User signs, presses "Capture sign" — the latest frame is sent to the
     recognizer, which returns a single token.
  3. Tokens accumulate in a visible list until the user presses "Speak",
     which composes them into an English sentence and synthesises voice.

A "Pose tracking" tab visualises the MediaPipe Holistic landmarks in
parallel — useful for the demo video to *show* the system isn't just an
LLM in a box.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import gradio as gr
import numpy as np

from signbridge.composer.sentence import compose_sentence
from signbridge.recognizer.landmarks import LandmarkExtractor
from signbridge.recognizer.vlm import recognize_sign_from_frame
from signbridge.voice.tts import synthesize_speech

logger = logging.getLogger(__name__)

RECOGNIZER_MODE = os.getenv("SIGNBRIDGE_RECOGNIZER_MODE", "vlm").lower()


def _sample_frames_from_video(video_path: str | None, n_frames: int = 4) -> list:
    """Open a video file and return n_frames evenly-spaced RGB frames.

    Returns [] if the file is missing or unreadable. Frames are RGB
    np.ndarray (HxWx3, uint8). Imports OpenCV lazily so the Gradio
    module still loads on machines without it.
    """
    if not video_path:
        return []
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "opencv-python-headless not installed; cannot sample video frames."
        )
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [
        int(round(i * (total - 1) / (n_frames - 1))) if n_frames > 1 else 0
        for i in range(n_frames)
    ]
    frames: list = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


@dataclass
class _SessionState:
    """Per-tab state. Gradio creates one per browser session."""

    sign_history: list[str] = field(default_factory=list)
    last_sentence: str = ""
    last_audio_path: str | None = None


# Single-user demo: one global latest-frame variable instead of a
# session-keyed dict. The previous session-keyed approach failed because
# adding `gr.Request` to a stream-handler signature appears to silently
# kill the handler in gradio 4.44.1 (TypeError swallowed by the queue
# worker). No request injection here = no failure surface.
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_stash_count = 0


def _new_session() -> _SessionState:
    return _SessionState()


def _format_history(signs: list[str]) -> str:
    if not signs:
        return "_(no signs captured yet — try signing the letter A and pressing Capture)_"
    return " · ".join(f"`{s}`" for s in signs)


def _recognize(frame: np.ndarray) -> tuple[str, float]:
    """Single-frame recognition for the Snapshot tab (fingerspelling).

    Tries the trained MediaPipe-Hand → MLP classifier first (88% accuracy
    on the holdout). Falls back to Qwen3-VL when the classifier is missing
    weights or MediaPipe can't detect a hand.
    """
    if RECOGNIZER_MODE == "classifier":
        from signbridge.recognizer.classifier import classify_landmarks

        extractor = _shared_extractor()
        _, landmarks = extractor.extract(frame)
        if landmarks is None:
            return "", 0.0
        return classify_landmarks(np.expand_dims(landmarks, axis=0))

    # Default 'vlm' mode — first try the landmark classifier, then VLM.
    from signbridge.recognizer.landmark_classifier import predict_letter

    token, conf = predict_letter(frame)
    if conf >= 0.5:
        return token, conf
    return recognize_sign_from_frame(frame)


_extractor_singleton: LandmarkExtractor | None = None
_extractor_lock = threading.Lock()


def _shared_extractor() -> LandmarkExtractor:
    """Return the lazy-loaded MediaPipe Holistic extractor.

    Double-checked locking so concurrent first-call requests under
    Gradio's worker threads don't race and build two extractors.
    """
    global _extractor_singleton
    if _extractor_singleton is not None:
        return _extractor_singleton
    with _extractor_lock:
        if _extractor_singleton is None:
            _extractor_singleton = LandmarkExtractor()
        return _extractor_singleton


def _stash_frame(frame: np.ndarray | None) -> None:
    """Webcam .change() callback — writes every live frame to the global
    `_latest_frame`. Bare signature (no gr.Request, no extra params) so
    gradio's signature inspection can't fail."""
    global _latest_frame, _stash_count
    if frame is None:
        return
    with _frame_lock:
        _latest_frame = frame
        _stash_count += 1
        # Log every ~30 frames (~1/sec at 30 fps webcam) so HF run logs
        # confirm the handler is actually firing.
        if _stash_count == 1 or _stash_count % 30 == 0:
            logger.info(
                "_stash_frame fired %d times; last shape=%s dtype=%s",
                _stash_count, frame.shape, frame.dtype,
            )


def _capture_sign(state: _SessionState) -> tuple[str, str, _SessionState]:
    """Take-image button handler. Reads the latest live frame from the
    global cache, runs recognition, appends to history."""
    with _frame_lock:
        frame = _latest_frame

    logger.info(
        "_capture_sign: frame_present=%s, stash_count=%d",
        frame is not None, _stash_count,
    )

    if frame is None:
        return (
            "_no frame yet — make sure the camera preview is live and try again_",
            _format_history(state.sign_history),
            state,
        )

    token, confidence = _recognize(frame)
    if not token or confidence < 0.5:
        return (
            "_couldn't recognise that one — try centering the gesture and a plain background_",
            _format_history(state.sign_history),
            state,
        )

    state.sign_history.append(token)
    return (
        f"detected: **{token}** ({confidence:.0%})",
        _format_history(state.sign_history),
        state,
    )


def _show_landmarks(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    annotated, _ = _shared_extractor().extract(frame)
    return annotated


def _speak(state: _SessionState) -> tuple[str, str | None, _SessionState]:
    if not state.sign_history:
        return "(no signs captured yet)", None, state

    sentence = compose_sentence(list(state.sign_history))
    state.last_sentence = sentence
    state.last_audio_path = synthesize_speech(sentence)
    return sentence, state.last_audio_path, state


def _clear(state: _SessionState) -> tuple[str, str, str, None, _SessionState]:
    """Reset history, sentence, and audio. Leave the streaming webcam
    intact so the next capture is one click away."""
    state.sign_history.clear()
    state.last_sentence = ""
    state.last_audio_path = None
    return (
        "",                                  # latest status text
        _format_history(state.sign_history), # history markdown
        "",                                  # composed sentence textbox
        None,                                # audio out
        state,
    )


_WEBCAM_BUTTON_LABEL_CSS = """
/* Gradio's gr.Image webcam shows an unlabelled webcam-icon (start) and
   red-square (stop) inside the preview. Add visible text labels via CSS
   pseudo-elements so first-time users know what each button does. */
.signbridge-webcam .source-selection .icon-with-text,
.signbridge-webcam button[aria-label*="webcam" i]::after,
.signbridge-webcam button[aria-label*="record" i]::after {
    content: " Start";
    margin-left: 6px;
    font-size: 14px;
    font-weight: 600;
    color: #4f46e5;
}
.signbridge-webcam button[aria-label*="stop" i]::after {
    content: " Stop";
    margin-left: 6px;
    font-size: 14px;
    font-weight: 600;
    color: #dc2626;
}
/* Make any webcam-control button render its aria-label as visible text. */
.signbridge-webcam .controls button {
    min-width: 80px;
}
/* Floating tooltip over the webcam pane on first load. */
.signbridge-webcam-help {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    padding: 8px 12px;
    margin: 6px 0 12px 0;
    border-radius: 6px;
    font-size: 13px;
    color: #1e1b4b;
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="SignBridge", theme=gr.themes.Soft(), css=_WEBCAM_BUTTON_LABEL_CSS
    ) as demo:
        gr.Markdown(
            "# 🤟 SignBridge — real-time ASL → English speech\n"
            "Two people who couldn't communicate, now can. **Snapshot** for "
            "fingerspelled letters; **Record sign** for full ASL words "
            "(motion-dependent). Powered by AMD Instinct MI300X."
        )

        # Pass the FACTORY (callable), not the result. Gradio invokes
        # callable State values once per session — guarantees per-tab
        # isolation. Using `gr.State(_new_session())` instead would create
        # a single shared instance at module-load time, which has been
        # observed to leak state across browser tabs in some Gradio 4.x
        # configurations.
        state = gr.State(_new_session)

        with gr.Tabs():
            with gr.Tab("Snapshot — fingerspelling"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML(
                            '<div class="signbridge-webcam-help">'
                            '<b>How it works:</b> '
                            '<b>1.</b> click the webcam once to grant access · '
                            '<b>2.</b> sign a letter (A–Z) · '
                            '<b>3.</b> click <b>📸 Take image</b> — recognition is automatic · '
                            '<b>4.</b> repeat for the next letter, then press <b>🔊 Speak</b>.'
                            "</div>"
                        )
                        webcam = gr.Image(
                            sources=["webcam"],
                            # streaming=True keeps the live preview running
                            # after the one-time permission grant, so the
                            # user never sees the access-prompt screen
                            # again. Frames are stashed in session state via
                            # the .stream() handler, and the Take-image
                            # button reads from there.
                            streaming=True,
                            label="Sign here",
                            height=420,
                            type="numpy",
                            elem_classes=["signbridge-webcam"],
                        )
                        with gr.Row():
                            capture_btn = gr.Button(
                                "📸 Take image", variant="primary", size="lg"
                            )
                            clear_btn = gr.Button(
                                "🧹 Clear", variant="secondary", size="lg"
                            )
                        latest = gr.Markdown(value="")

                    with gr.Column(scale=2):
                        history = gr.Markdown(
                            value=_format_history([]), label="Captured signs"
                        )
                        speak_btn = gr.Button("🔊 Speak", variant="primary", size="lg")
                        sentence_box = gr.Textbox(
                            label="Composed sentence",
                            interactive=False,
                            lines=3,
                        )
                        audio_out = gr.Audio(label="Spoken response", autoplay=True)
                        gr.Markdown(
                            "**Tip:** for V1, try the ASL fingerspelling alphabet (A–Z, 0–9) "
                            "or one of the WLASL Top-50 signs (`hello`, `thank_you`, `name`, "
                            "`please`, `sorry`, `family`, `eat`, `drink`, `home`, `love`, …). "
                            "Spell out a word letter-by-letter, then press Speak."
                        )

                # Use .change() (not .stream()) — it fires on every value
                # change which, for a streaming webcam, is every frame.
                # outputs=[] is required (None doesn't wire the event).
                webcam.change(
                    fn=_stash_frame,
                    inputs=[webcam],
                    outputs=[],
                    show_progress="hidden",
                )
                capture_btn.click(
                    fn=_capture_sign,
                    inputs=[state],
                    outputs=[latest, history, state],
                )
                speak_btn.click(
                    fn=_speak,
                    inputs=[state],
                    outputs=[sentence_box, audio_out, state],
                )
                clear_btn.click(
                    fn=_clear,
                    inputs=[state],
                    outputs=[latest, history, sentence_box, audio_out, state],
                )

            with gr.Tab("Record sign — full ASL words"):
                gr.Markdown(
                    "Record 1.5–2 s of yourself signing a full ASL word "
                    "(`hello`, `thank_you`, `please`, `eat`, `drink`, …). "
                    "The recognizer samples 4 frames from the clip and uses "
                    "motion across them to decide."
                )
                gr.HTML(
                    '<div class="signbridge-webcam-help">'
                    '<b>How it works:</b> '
                    '<b>1.</b> click the webcam to access it · '
                    '<b>2.</b> click <b>● Record</b>, sign for 1.5–2 seconds, click <b>■ Stop</b> · '
                    '<b>3.</b> press <b>🎬 Submit recording</b> below.'
                    "</div>"
                )
                video_in = gr.Video(
                    sources=["webcam"],
                    label="Hold while signing",
                    height=420,
                    elem_classes=["signbridge-webcam"],
                )
                with gr.Row():
                    submit_video_btn = gr.Button(
                        "🎬 Submit recording",
                        variant="primary",
                        size="lg",
                    )
                video_status = gr.Markdown(value="")

                def _handle_video(
                    video_path: str | None, sess: _SessionState
                ) -> tuple[str, str, _SessionState]:
                    frames = _sample_frames_from_video(video_path, n_frames=4)
                    if len(frames) < 2:
                        return (
                            "_couldn't read enough frames — try recording again_",
                            _format_history(sess.sign_history),
                            sess,
                        )
                    from signbridge.recognizer.vlm import recognize_sign_from_frames

                    token, confidence = recognize_sign_from_frames(frames)
                    if not token or confidence < 0.5:
                        return (
                            "_couldn't recognise that one — try slower, plain background_",
                            _format_history(sess.sign_history),
                            sess,
                        )
                    sess.sign_history.append(token)
                    return (
                        f"detected: **{token}** ({confidence:.0%})",
                        _format_history(sess.sign_history),
                        sess,
                    )

                submit_video_btn.click(
                    fn=_handle_video,
                    inputs=[video_in, state],
                    outputs=[video_status, history, state],
                )

        with gr.Accordion("System info", open=False):
            gr.Markdown(
                f"- **Recognizer mode:** `{RECOGNIZER_MODE}` "
                f"({'VLM via OpenAI-compatible endpoint' if RECOGNIZER_MODE == 'vlm' else 'trained landmark classifier'})\n"
                f"- **Provider:** `{os.getenv('SIGNBRIDGE_PROVIDER', 'amd')}` "
                f"(set `SIGNBRIDGE_PROVIDER=openai|hf|amd` in `.env`)\n"
                f"- **Composer model:** `{os.getenv('SIGNBRIDGE_COMPOSER_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')}`\n"
                f"- **TTS model:** `{os.getenv('SIGNBRIDGE_TTS_MODEL', 'tts_models/multilingual/multi-dataset/xtts_v2')}`\n"
            )

    return demo
