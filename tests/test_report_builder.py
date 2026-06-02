"""Tests for the single-analysis report builder."""
from __future__ import annotations

from report_builder import (
    build_analysis_dataframe,
    build_markdown_report,
    report_to_bytes,
)


RESULTS = [
    {
        "domain": "Privacy & Security",
        "subdomain": "Compromise of privacy",
        "definition": "Sensitive data exposure.",
        "score": 0.623456,
    },
    {
        "domain": "Discrimination",
        "subdomain": "Unfair discrimination",
        "definition": "Biased outcomes.",
        "score": 0.551234,
    },
]

RATIONALE = {
    "summary": "The project handles sensitive data and automated decisions.",
    "rationales": [
        {"subdomain": "Compromise of privacy", "rationale": "It infers health data."},
    ],
}


def test_build_analysis_dataframe_shape_and_rounding():
    df = build_analysis_dataframe(RESULTS)
    assert list(df.columns) == ["rank", "domain", "subdomain", "cosine_score"]
    assert len(df) == 2
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[0]["cosine_score"] == 0.6235


def test_build_analysis_dataframe_empty():
    df = build_analysis_dataframe([])
    assert list(df.columns) == ["rank", "domain", "subdomain", "cosine_score"]
    assert len(df) == 0


def test_markdown_report_includes_matches_and_rationale():
    md = build_markdown_report("A health prediction app.", RESULTS, RATIONALE)
    assert "# AI Risk Analysis Report" in md
    assert "A health prediction app." in md
    assert "Compromise of privacy" in md
    assert "0.623" in md
    assert "## Summary" in md
    assert "It infers health data." in md


def test_markdown_report_without_rationale_omits_sections():
    md = build_markdown_report("A health prediction app.", RESULTS, None)
    assert "## Top matching risks" in md
    assert "## Summary" not in md
    assert "## Why these risks apply" not in md


def test_report_to_bytes_round_trips():
    md = build_markdown_report("desc", RESULTS, None)
    raw = report_to_bytes(md)
    assert isinstance(raw, bytes)
    assert raw.decode("utf-8").startswith("# AI Risk Analysis Report")
