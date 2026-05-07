"""Tests for the sentence composer fallback path (no API keys configured)."""

from __future__ import annotations

import pytest

from signbridge.composer.sentence import _naive_join, _strip_quotes, compose_sentence


class TestNaiveJoin:
    def test_empty(self) -> None:
        assert _naive_join([]) == ""

    def test_single_lowercase_gloss(self) -> None:
        assert _naive_join(["hello"]) == "Hello."

    def test_single_letter(self) -> None:
        assert _naive_join(["A"]) == "A."

    def test_fingerspelled_word(self) -> None:
        assert _naive_join(["L", "U", "C", "A", "S"]) == "Lucas."

    def test_fingerspelled_then_gloss(self) -> None:
        # "L U C A S" + "hello" → "Lucas hello."
        assert _naive_join(["L", "U", "C", "A", "S", "hello"]) == "Lucas hello."

    def test_gloss_with_underscore(self) -> None:
        # thank_you should render as "thank you"
        assert "thank you" in _naive_join(["thank_you"]).lower()

    def test_existing_punctuation_not_doubled(self) -> None:
        # Naive join always appends '.', but never twice.
        out = _naive_join(["hello"])
        assert out.count(".") == 1


class TestStripQuotes:
    def test_no_quotes(self) -> None:
        assert _strip_quotes("hello") == "hello"

    def test_double_quotes(self) -> None:
        assert _strip_quotes('"hello"') == "hello"

    def test_single_quotes(self) -> None:
        assert _strip_quotes("'hello'") == "hello"

    def test_internal_quotes_preserved(self) -> None:
        assert _strip_quotes('"he said \'hi\'"') == "he said 'hi'"


class TestComposeSentenceFallback:
    """Without API keys, compose should fall back to _naive_join."""

    def test_empty(self) -> None:
        assert compose_sentence([]) == ""

    def test_falls_back_to_naive(self) -> None:
        # No AMD/OpenAI keys → falls back to _naive_join
        result = compose_sentence(["H", "I"])
        assert result == "Hi."

    def test_falls_back_for_glosses(self) -> None:
        result = compose_sentence(["hello", "name", "L", "U", "C", "A", "S"])
        assert "Lucas" in result
        assert "hello" in result.lower()
        assert "name" in result.lower()


class TestComposeSentenceWithMockProvider:
    """When an API key is set we ATTEMPT to call the provider, but on failure
    we still fall back. We simulate a working provider by monkeypatching the
    OpenAI client."""

    def test_provider_returns_clean_sentence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNBRIDGE_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Stub OpenAI client at the module level
        from signbridge.composer import sentence as sentence_mod

        class _FakeChoice:
            def __init__(self, content: str) -> None:
                self.message = type("M", (), {"content": content})()

        class _FakeResp:
            def __init__(self, content: str) -> None:
                self.choices = [_FakeChoice(content)]

        class _FakeChat:
            def __init__(self, content: str) -> None:
                self._content = content
                self.completions = self

            def create(self, **_: object) -> _FakeResp:
                return _FakeResp(self._content)

        class _FakeClient:
            def __init__(self, content: str) -> None:
                self.chat = _FakeChat(content)

        monkeypatch.setattr(
            sentence_mod,
            "_resolve_client",
            lambda: (_FakeClient("My name is Lucas."), "test-model"),
        )
        out = compose_sentence(["name", "L", "U", "C", "A", "S"])
        assert out == "My name is Lucas."

    def test_provider_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.composer import sentence as sentence_mod

        class _FailingClient:
            class _FailingChat:
                completions = type(
                    "_C",
                    (),
                    {"create": staticmethod(lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))},
                )()
            chat = _FailingChat()

        monkeypatch.setattr(
            sentence_mod, "_resolve_client", lambda: (_FailingClient(), "test")
        )
        out = compose_sentence(["hello"])
        assert out == "Hello."  # fell back to naive
