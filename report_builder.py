"""Builds downloadable reports for a single project analysis.

Turns the in-memory analysis (the user description, the top vector matches, and
the optional LLM rationale) into a flat table for CSV export and a readable
Markdown report. The Streamlit UI offers both as downloads so a single analysis
can be saved or shared, mirroring the existing bulk CSV export.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def build_analysis_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Flatten the top matches into a ranked table for CSV export."""
    rows = []
    for rank, r in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "domain": r["domain"],
                "subdomain": r["subdomain"],
                "cosine_score": round(float(r["score"]), 4),
            }
        )
    return pd.DataFrame(rows, columns=["rank", "domain", "subdomain", "cosine_score"])


def build_markdown_report(
    user_input: str,
    results: List[Dict],
    rationale: Optional[Dict] = None,
) -> str:
    """Build a readable Markdown report for one analysis.

    Includes the project description, a ranked matches table, and, when a
    rationale is available, the summary and per-risk explanations.
    """
    lines: List[str] = ["# AI Risk Analysis Report", ""]

    lines += ["## Project description", "", user_input.strip(), ""]

    lines += ["## Top matching risks", ""]
    lines += ["| Rank | Domain | Subdomain | Cosine score |", "| --- | --- | --- | --- |"]
    for rank, r in enumerate(results, start=1):
        lines.append(
            f"| {rank} | {r['domain']} | {r['subdomain']} | {float(r['score']):.3f} |"
        )
    lines.append("")

    if rationale:
        summary = rationale.get("summary", "")
        if summary:
            lines += ["## Summary", "", summary, ""]

        entries = rationale.get("rationales", [])
        if entries:
            lines += ["## Why these risks apply", ""]
            for item in entries:
                sub = item.get("subdomain", "") or "Risk"
                text = item.get("rationale", "")
                lines += [f"### {sub}", "", text, ""]

    return "\n".join(lines).rstrip() + "\n"


def report_to_bytes(text: str) -> bytes:
    """Encode a text report as UTF-8 bytes for a Streamlit download button."""
    return text.encode("utf-8")


if __name__ == "__main__":
    demo_results = [
        {"domain": "Privacy & Security", "subdomain": "Compromise of privacy", "score": 0.62},
        {"domain": "Discrimination", "subdomain": "Unfair discrimination", "score": 0.55},
    ]
    demo_rationale = {
        "summary": "The project handles sensitive data and automated decisions.",
        "rationales": [
            {"subdomain": "Compromise of privacy", "rationale": "It infers health data."},
        ],
    }
    print(build_analysis_dataframe(demo_results).to_string(index=False))
    print()
    print(build_markdown_report("A health prediction app.", demo_results, demo_rationale))
