"""Tests for the TTS module — silent-stub fallback path."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from signbridge.voice.tts import synthesize_speech


class TestSynthesize:
    def test_empty_text_returns_none(self) -> None:
        assert synthesize_speech("") is None

    def test_returns_path_to_existing_wav(self) -> None:
        # Without the heavy Coqui XTTS package, the module emits a silent stub.
        # That stub MUST be a real readable WAV file.
        path = synthesize_speech("hello, world!")
        assert path is not None and path != ""
        p = Path(path)
        assert p.exists()
        assert p.suffix == ".wav"

        # Read it back to confirm it's a valid WAV
        data, sr = sf.read(str(p))
        assert sr > 0
        assert len(data) > 0

    def test_caches_per_text(self) -> None:
        path1 = synthesize_speech("repeat me")
        path2 = synthesize_speech("repeat me")
        assert path1 == path2

    def test_different_text_different_file(self) -> None:
        path1 = synthesize_speech("alpha")
        path2 = synthesize_speech("beta")
        assert path1 != path2
