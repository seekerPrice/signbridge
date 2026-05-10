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
            print("[recognize] holistic: no landmarks detected", flush=True)
            return "", 0.0
        token, conf = classify_landmarks(np.expand_dims(landmarks, axis=0))
        print(f"[recognize] holistic-classifier: token={token!r} conf={conf:.2f}", flush=True)
        return token, conf

    # Default 'vlm' mode — first try the landmark classifier, then VLM.
    from signbridge.recognizer.landmark_classifier import predict_letter

    token, conf = predict_letter(frame)
    print(f"[recognize] mediapipe+MLP: token={token!r} conf={conf:.2f}", flush=True)
    if conf >= 0.5:
        return token, conf
    print("[recognize] MLP below threshold; falling through to VLM", flush=True)
    vtoken, vconf = recognize_sign_from_frame(frame)
    print(f"[recognize] VLM result: token={vtoken!r} conf={vconf:.2f}", flush=True)
    return vtoken, vconf


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


_MIN_CONF_ACCEPT = 0.75   # ≥ this → token accepted into history; below → shown as ✗ dropped


def _on_snapshot(
    frame: np.ndarray | None, state: _SessionState
) -> tuple[str, str, _SessionState, "gr.components.Image"]:
    """Webcam .change() callback. Fires once per user snapshot in
    non-streaming mode. Recognises the frame, appends to history,
    then returns gr.update(value=None) so the Webcam re-mounts and
    _AUTO_ACCESS_WEBCAM_JS auto-clicks the access placeholder.

    Bounce guard: when we set value=None below, gradio dispatches
    another .change() with frame=None. The first branch makes that
    a no-op so we don't loop forever."""
    print(
        f"[snapshot] frame_present={frame is not None}"
        + (f" shape={frame.shape}" if frame is not None else ""),
        flush=True,
    )

    if frame is None:
        return ("", _format_history(state.sign_history), state, gr.update())

    # Defensive downscale: even if the JS resolution cap didn't apply
    # (e.g. older browser), keep the recognition path on a manageable
    # frame. MediaPipe + MLP inference is faster on smaller input too.
    h, w = frame.shape[:2]
    if max(h, w) > 720:
        try:
            from PIL import Image as PILImage
            scale = 720 / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            frame = np.array(
                PILImage.fromarray(frame).resize(new_size, PILImage.BILINEAR)
            )
            print(f"[snapshot] downscaled to {frame.shape}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[snapshot] downscale failed: {exc}", flush=True)

    token, confidence = _recognize(frame)
    print(f"[snapshot] recognised token={token!r} conf={confidence:.2f}", flush=True)

    from signbridge.recognizer import landmark_classifier as lc
    top3 = list(lc.last_top3)
    top3_str = ", ".join(f"`{t}` ({c:.0%})" for t, c in top3) if top3 else ""

    def _html_box(border_left: str, body: str) -> str:
        return (
            f'<div style="background:#f8fafc;border:1px solid #cbd5e1;'
            f'border-left:4px solid {border_left};border-radius:6px;'
            f'padding:12px 16px;margin:8px 0;min-height:56px;font-size:15px;">'
            f'{body}</div>'
        )

    top3_html = ", ".join(
        f"<code>{t}</code> ({c:.0%})" for t, c in top3
    ) if top3 else ""

    if not token:
        msg = _html_box(
            "#dc2626",
            "<b>✗ no hand detected</b> — show your hand clearly in frame and try again.",
        )
        return (msg, _format_history(state.sign_history), state, gr.update(value=None))

    if confidence < _MIN_CONF_ACCEPT:
        body = (
            f"<b style='color:#dc2626'>✗ dropped <code>{token}</code> "
            f"({confidence:.0%})</b> — too uncertain, please re-sign with a clearer pose."
        )
        if top3_html:
            body += f"<br><span style='color:#475569;font-size:14px'>alternatives: {top3_html}</span>"
        return (_html_box("#dc2626", body), _format_history(state.sign_history), state, gr.update(value=None))

    state.sign_history.append(token)
    body = (
        f"<b style='color:#16a34a'>✓ added <code>{token}</code> "
        f"({confidence:.0%})</b>"
    )
    if top3_html:
        body += f"<br><span style='color:#475569;font-size:14px'>alternatives: {top3_html}</span>"
    return (_html_box("#16a34a", body), _format_history(state.sign_history), state, gr.update(value=None))


def _show_landmarks(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    annotated, _ = _shared_extractor().extract(frame)
    return annotated


def _speak(state: _SessionState) -> tuple[str, str | None, _SessionState]:
    if not state.sign_history:
        print("[speak] no signs to compose; returning empty.", flush=True)
        return "(no signs captured yet)", None, state

    print(f"[speak] composing from {len(state.sign_history)} tokens: {state.sign_history}", flush=True)
    sentence = compose_sentence(list(state.sign_history))
    print(f"[speak] composed sentence: {sentence!r}", flush=True)
    state.last_sentence = sentence
    print("[speak] synthesising speech...", flush=True)
    state.last_audio_path = synthesize_speech(sentence)
    print(f"[speak] audio_path={state.last_audio_path}", flush=True)
    return sentence, state.last_audio_path, state


_LATEST_PLACEHOLDER_HTML = (
    '<div style="background:#f8fafc;border:1px solid #cbd5e1;'
    'border-left:4px solid #4f46e5;border-radius:6px;'
    'padding:12px 16px;margin:8px 0;min-height:56px;font-size:15px;">'
    '<i>(awaiting capture — click the 📷 camera button above)</i>'
    '</div>'
)


def _clear(state: _SessionState) -> tuple[str, str, str, None, _SessionState]:
    """Reset history, sentence, and audio. Leave the streaming webcam
    intact so the next capture is one click away."""
    state.sign_history.clear()
    state.last_sentence = ""
    state.last_audio_path = None
    return (
        _LATEST_PLACEHOLDER_HTML,            # latest status (HTML)
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
/* Snapshot tab uses gradio's built-in snapshot camera button as the
   sole capture trigger. Streaming had to be dropped because HF Space's
   proxy can't sustain the per-500ms upload rate. Source-select dropdown
   is hidden to keep the UI clean. */
.signbridge-webcam-snapshot .source-selection {
    display: none !important;
}
/* Make the per-click status banner (latest = gr.Markdown) impossible to
   miss — bordered box with a heading-like label so users see ✓ added /
   ✗ dropped right under the camera. */
#signbridge-latest-status {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #4f46e5;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    min-height: 56px;
    font-size: 15px;
}
#signbridge-latest-status p {
    margin: 0;
}
"""


# JS injected at app load. Runs in the browser.
#
# Non-streaming gr.Image webcam unmounts the Webcam component each time
# the value clears (gradio's ImageUploader.svelte: shows the captured
# image when value!=null, shows Webcam only when value==null). Each
# remount re-renders the "Click to Access Webcam" placeholder. After
# the first user-gesture grant, the browser remembers permission, so
# we can programmatically click that placeholder to snap straight back
# to live preview — making per-letter UX a single click on the camera
# button instead of click-allow-then-camera.
_AUTO_ACCESS_WEBCAM_JS = """
() => {
    // 1) Cap webcam resolution at 640x480. Gradio 4.44.1 hardcodes
    //    {ideal: 1920x1440}, which on HF Spaces produces ~1-2MB PNG
    //    uploads per click — slow enough to hit ClientDisconnect on
    //    HF's proxy. A 640x480 frame is ~50-100KB and uploads in <1s.
    //    Monkey-patch getUserMedia BEFORE Gradio's Webcam.svelte calls it.
    const origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(
        navigator.mediaDevices
    );
    navigator.mediaDevices.getUserMedia = (constraints) => {
        if (constraints && constraints.video) {
            const v = constraints.video;
            const newVideo =
                typeof v === 'object'
                    ? { ...v, width: { ideal: 640 }, height: { ideal: 480 } }
                    : { width: { ideal: 640 }, height: { ideal: 480 } };
            constraints = { ...constraints, video: newVideo };
            console.log('[signbridge] capped webcam resolution at 640x480');
        }
        return origGetUserMedia(constraints);
    };

    // 2) After each capture, gradio's Webcam.svelte unmounts and remounts
    //    the access placeholder. Auto-click it so per-letter UX is just
    //    one click on the camera button. Wait 1.5s before clicking so
    //    the in-flight upload XHR has time to actually establish — too
    //    fast a re-click was racing the upload and causing ClientDisconnect.
    const SELECTOR = '.signbridge-webcam-snapshot button[title="grant webcam access" i], .signbridge-webcam-snapshot div[title="grant webcam access" i] button';
    let firstGrantSeen = false;
    let lastAutoClickAt = 0;
    const tick = () => {
        document.querySelectorAll(SELECTOR).forEach((btn) => {
            if (btn.dataset.signbridgeAutoaccessed) return;
            if (!firstGrantSeen) {
                firstGrantSeen = true;
                return;
            }
            const now = Date.now();
            if (now - lastAutoClickAt < 1500) return;
            btn.click();
            btn.dataset.signbridgeAutoaccessed = '1';
            lastAutoClickAt = now;
            console.log('[signbridge] auto-accessed re-mounted webcam');
        });
    };
    setInterval(tick, 500);
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="SignBridge",
        theme=gr.themes.Soft(),
        css=_WEBCAM_BUTTON_LABEL_CSS,
        js=_AUTO_ACCESS_WEBCAM_JS,
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
                            '<b>1.</b> click the preview once to grant camera access (one-time) · '
                            '<b>2.</b> sign a letter (A–Z) · '
                            '<b>3.</b> click the <b>📷 camera button</b> in the preview — recognition is automatic, then the preview re-arms · '
                            '<b>4.</b> repeat for the next letter, then press <b>🔊 Speak</b>.'
                            "</div>"
                        )
                        # streaming=True was deployable locally but HF
                        # Space's proxy can't sustain the per-500ms
                        # frame uploads — every upload_file POST hit
                        # ClientDisconnect, no frame ever reached
                        # Python. Switching to non-streaming snapshot
                        # mode: one upload per click, reliable on HF.
                        # The webcam re-mounts after auto-clear; the
                        # _AUTO_ACCESS_WEBCAM_JS injected at app load
                        # re-clicks the access placeholder so per-letter
                        # UX stays a single click on gradio's snapshot
                        # camera button (no double-grant per letter).
                        webcam = gr.Image(
                            sources=["webcam"],
                            label="Sign here — click the 📷 camera button",
                            height=420,
                            type="numpy",
                            elem_classes=["signbridge-webcam", "signbridge-webcam-snapshot"],
                        )
                        with gr.Row():
                            clear_btn = gr.Button(
                                "🧹 Clear history", variant="secondary", size="lg"
                            )
                        latest = gr.HTML(value=_LATEST_PLACEHOLDER_HTML)

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

                # In non-streaming mode, .change() fires once per user
                # snapshot (camera button click). We get the frame
                # directly as the input — no global cache or stash
                # plumbing needed. Auto-clear the value at the end so
                # gradio re-mounts the Webcam component, which together
                # with _AUTO_ACCESS_WEBCAM_JS makes per-letter UX one
                # click.
                webcam.change(
                    fn=_on_snapshot,
                    inputs=[webcam, state],
                    outputs=[latest, history, state, webcam],
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
