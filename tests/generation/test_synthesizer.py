"""
Contract + adversarial tests for Phase 2 Gemini Synthesis (generation/synthesizer.py).

Drives Synthesizer with a fake google-genai-SDK-shaped client (no network, no `google.genai`
import) covering: immediate success, transient-then-success, persistent failure (fail-closed,
never raises), retry/backoff timing, and the exact call parameters sent to the client.
"""

from __future__ import annotations

import pytest

from generation.synthesizer import (
    GEMINI_MODEL,
    INSUFFICIENT_DATA_MARKER,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    Synthesizer,
)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, behaviors):
        # `behaviors` is a list of callables/values consumed one per call; a value that's an
        # Exception instance is raised instead of returned.
        self._behaviors = list(behaviors)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return _FakeResponse(behavior)


class _FakeGeminiClient:
    def __init__(self, behaviors):
        self.models = _FakeModels(behaviors)


class _RecordingSleep:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float):
        self.calls.append(seconds)


class TestImmediateSuccess:
    def test_returns_response_text(self):
        client = _FakeGeminiClient(["CLAIM: x\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: high\n---"])
        synthesizer = Synthesizer(client=client, sleep_fn=_RecordingSleep())
        result = synthesizer.synthesize("prompt text")
        assert result == "CLAIM: x\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: high\n---"

    def test_calls_client_with_expected_settings(self):
        client = _FakeGeminiClient(["ok"])
        synthesizer = Synthesizer(client=client, sleep_fn=_RecordingSleep())
        synthesizer.synthesize("my prompt")
        call = client.models.calls[0]
        assert call["model"] == GEMINI_MODEL == "gemini-flash-latest"
        assert call["contents"] == "my prompt"
        assert call["config"]["temperature"] == TEMPERATURE == 0.3
        assert call["config"]["max_output_tokens"] == MAX_OUTPUT_TOKENS == 800

    def test_no_sleep_on_immediate_success(self):
        client = _FakeGeminiClient(["ok"])
        sleep = _RecordingSleep()
        Synthesizer(client=client, sleep_fn=sleep).synthesize("p")
        assert sleep.calls == []


class TestTransientThenSuccess:
    def test_retries_and_returns_eventual_success(self):
        client = _FakeGeminiClient([ConnectionError("timeout"), "recovered text"])
        sleep = _RecordingSleep()
        result = Synthesizer(client=client, sleep_fn=sleep).synthesize("p")
        assert result == "recovered text"
        assert len(client.models.calls) == 2

    def test_backoff_is_exponential(self):
        client = _FakeGeminiClient([ConnectionError("a"), ConnectionError("b"), "recovered"])
        sleep = _RecordingSleep()
        Synthesizer(
            client=client, base_backoff_seconds=1.0, max_attempts=3, sleep_fn=sleep
        ).synthesize("p")
        assert sleep.calls == [1.0, 2.0]

    def test_custom_base_backoff_respected(self):
        client = _FakeGeminiClient([TimeoutError("a"), "ok"])
        sleep = _RecordingSleep()
        Synthesizer(client=client, base_backoff_seconds=0.5, sleep_fn=sleep).synthesize("p")
        assert sleep.calls == [0.5]


class TestPersistentFailureFailsClosed:
    def test_never_raises_returns_marker(self):
        client = _FakeGeminiClient([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        sleep = _RecordingSleep()
        result = Synthesizer(client=client, max_attempts=3, sleep_fn=sleep).synthesize("p")
        assert result == INSUFFICIENT_DATA_MARKER

    def test_exhausts_exactly_max_attempts(self):
        client = _FakeGeminiClient([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        Synthesizer(client=client, max_attempts=3, sleep_fn=_RecordingSleep()).synthesize("p")
        assert len(client.models.calls) == 3

    def test_sleeps_between_attempts_but_not_after_final_failure(self):
        client = _FakeGeminiClient([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        sleep = _RecordingSleep()
        Synthesizer(client=client, max_attempts=3, sleep_fn=sleep).synthesize("p")
        assert len(sleep.calls) == 2  # 2 sleeps between 3 attempts, none after the last

    def test_max_attempts_one_means_no_retry_at_all(self):
        client = _FakeGeminiClient([RuntimeError("only try")])
        sleep = _RecordingSleep()
        result = Synthesizer(client=client, max_attempts=1, sleep_fn=sleep).synthesize("p")
        assert result == INSUFFICIENT_DATA_MARKER
        assert len(client.models.calls) == 1
        assert sleep.calls == []

    def test_marker_is_not_valid_claim_format(self):
        # Structural guarantee that Phase 3's parser will yield zero claims from this marker.
        assert "CLAIM:" not in INSUFFICIENT_DATA_MARKER
        assert "SOURCE_CHUNK_ID:" not in INSUFFICIENT_DATA_MARKER


class TestHostileExceptionTypes:
    def test_non_standard_exception_subclass_still_retried(self):
        class WeirdSdkError(Exception):
            pass

        client = _FakeGeminiClient([WeirdSdkError("weird"), "recovered"])
        result = Synthesizer(client=client, sleep_fn=_RecordingSleep()).synthesize("p")
        assert result == "recovered"

    def test_response_with_no_text_attribute_fails_closed_not_crash(self):
        class _NoTextResponse:
            pass

        class _BrokenModels:
            def generate_content(self, **kwargs):
                return _NoTextResponse()

        class _BrokenClient:
            def __init__(self):
                self.models = _BrokenModels()

        result = Synthesizer(
            client=_BrokenClient(), max_attempts=2, sleep_fn=_RecordingSleep()
        ).synthesize("p")
        assert result == INSUFFICIENT_DATA_MARKER


class TestEmptyAndExtremeInputs:
    def test_empty_prompt_string(self):
        client = _FakeGeminiClient(["ok"])
        result = Synthesizer(client=client, sleep_fn=_RecordingSleep()).synthesize("")
        assert result == "ok"
        assert client.models.calls[0]["contents"] == ""

    def test_very_long_prompt_does_not_crash(self):
        client = _FakeGeminiClient(["ok"])
        prompt = "x " * 100_000
        result = Synthesizer(client=client, sleep_fn=_RecordingSleep()).synthesize(prompt)
        assert result == "ok"

    def test_empty_response_text_passed_through(self):
        client = _FakeGeminiClient([""])
        result = Synthesizer(client=client, sleep_fn=_RecordingSleep()).synthesize("p")
        assert result == ""


class TestClientResolution:
    def test_missing_api_key_env_var_raises_when_no_client_injected(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        synthesizer = Synthesizer()  # no client injected -> tries real lazy resolution
        with pytest.raises((KeyError, ImportError, ModuleNotFoundError)):
            synthesizer.synthesize("p")
