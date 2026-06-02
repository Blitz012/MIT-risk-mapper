"""Tests for the rationale engine.

The Ollama call is faked with a stub OpenAI client, so these run offline. Error
paths that trigger before any network call are tested directly.
"""
from __future__ import annotations

import types

import pytest

import llm_evaluator
from llm_evaluator import RationaleError, _build_messages, generate_rationale


SAMPLE_MATCHES = [
    {
        "domain": "Privacy & Security",
        "subdomain": "Compromise of privacy",
        "definition": "AI that collects or infers sensitive personal data.",
        "score": 0.62,
    }
]


def test_empty_input_raises():
    with pytest.raises(RationaleError):
        generate_rationale("   ", SAMPLE_MATCHES)


def test_no_matches_raises():
    with pytest.raises(RationaleError):
        generate_rationale("a health prediction app", [])


def test_build_messages_grounds_in_taxonomy():
    messages = _build_messages("a health app", SAMPLE_MATCHES)
    assert [m["role"] for m in messages] == ["system", "user"]
    user_content = messages[1]["content"]
    assert "Compromise of privacy" in user_content
    assert "Privacy & Security" in user_content
    assert "sensitive personal data" in user_content
    assert "0.620" in user_content


def _install_fake_openai(monkeypatch, content):
    """Patch openai.OpenAI with a stub whose response returns the given content."""

    def create(**kwargs):
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake_module = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)


def test_happy_path_parses_rationales(monkeypatch):
    payload = (
        '{"summary": "Overall risk summary.", '
        '"rationales": [{"subdomain": "Compromise of privacy", '
        '"rationale": "It infers health data."}]}'
    )
    _install_fake_openai(monkeypatch, payload)

    result = generate_rationale("a health app", SAMPLE_MATCHES)
    assert result["summary"] == "Overall risk summary."
    assert result["rationales"][0]["subdomain"] == "Compromise of privacy"


def test_invalid_json_raises_rationale_error(monkeypatch):
    _install_fake_openai(monkeypatch, "this is not json")
    with pytest.raises(RationaleError):
        generate_rationale("a health app", SAMPLE_MATCHES)
