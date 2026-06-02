"""LLM Rationale Engine for the MIT AI Risk Mapping Tool.

Takes the user's project description plus the top vector matches from the
risk mapper and asks a local Ollama model to explain, in structured form, why
each matched risk applies. The vector layer stays the source of truth: this
module only adds human readable justification on top of it.

Ollama exposes an OpenAI compatible endpoint, so we reuse the OpenAI SDK and
simply point it at the local server. No API key is required: the key field is
a placeholder that Ollama ignores.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


class RationaleError(Exception):
    """Raised when a rationale cannot be generated (missing key or API failure)."""


def _build_messages(user_input: str, matches: List[Dict]) -> List[Dict[str, str]]:
    """Construct the chat messages that ground the model in the taxonomy."""
    match_lines = []
    for i, m in enumerate(matches, start=1):
        match_lines.append(
            f"{i}. Domain: {m['domain']}\n"
            f"   Subdomain: {m['subdomain']}\n"
            f"   Taxonomy definition: {m['definition']}\n"
            f"   Cosine similarity: {m['score']:.3f}"
        )
    matches_block = "\n".join(match_lines)

    system_prompt = (
        "You are an AI risk evaluator. You are given a project description and "
        "the closest matching risks from the MIT AI Risk taxonomy, each with its "
        "official definition. For every match, explain concisely why that risk "
        "applies to this specific project, citing concrete elements of the "
        "description and grounding your reasoning in the provided definition. "
        "Do not invent risks that are not in the list. Keep each rationale to two "
        "or three sentences. Do not use em dashes or en dashes anywhere in your "
        "output."
    )

    user_prompt = (
        f"Project description:\n{user_input}\n\n"
        f"Top matching risks:\n{matches_block}\n\n"
        "Return a JSON object with this exact shape:\n"
        "{\n"
        '  "summary": "one sentence overall risk summary",\n'
        '  "rationales": [\n'
        '    {"subdomain": "<subdomain text>", "rationale": "<why it applies>"}\n'
        "  ]\n"
        "}\n"
        "Provide one rationale entry per matching risk, in the same order."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_rationale(
    user_input: str,
    matches: List[Dict],
    model: str = DEFAULT_MODEL,
) -> Dict:
    """Generate a structured rationale for the given matches.

    Returns a dict shaped like:
        {"summary": str, "rationales": [{"subdomain": str, "rationale": str}, ...]}

    Raises RationaleError if the local Ollama server cannot be reached or the
    call fails, so the caller can degrade gracefully and still show the vector
    results.
    """
    if not user_input or not user_input.strip():
        raise RationaleError("Cannot generate a rationale for empty input.")
    if not matches:
        raise RationaleError("No matches were provided to explain.")

    # Imported here so the module loads even if the SDK is not installed yet.
    from openai import OpenAI

    # Ollama ignores the api_key, but the SDK requires a non-empty value.
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    messages = _build_messages(user_input, matches)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RationaleError("The model returned a response that was not valid JSON.") from exc
    except Exception as exc:
        raise RationaleError(
            "The rationale request failed. Make sure Ollama is running "
            f"('ollama serve') and the model is pulled ('ollama pull {model}'). "
            f"Details: {exc}"
        ) from exc

    rationales = parsed.get("rationales", [])
    if not isinstance(rationales, list):
        raise RationaleError("The model response did not contain a rationale list.")

    return {
        "summary": parsed.get("summary", ""),
        "rationales": rationales,
    }


if __name__ == "__main__":
    sample_matches = [
        {
            "domain": "Privacy & Security",
            "subdomain": "Compromise of privacy",
            "definition": "AI systems that collect or infer sensitive personal data.",
            "score": 0.62,
        }
    ]
    try:
        result = generate_rationale(
            "A health app that predicts patient outcomes from medical records.",
            sample_matches,
        )
        print(json.dumps(result, indent=2))
    except RationaleError as err:
        print(f"Rationale unavailable: {err}")
