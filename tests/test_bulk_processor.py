"""Tests for bulk CSV processing.

A fake mapper stands in for RiskMapper so these tests stay fast and do not load
the real embedding model.
"""
from __future__ import annotations

import pandas as pd
import pytest

from bulk_processor import process_descriptions, to_csv_bytes
from nlp_mapper import RiskEntry


class FakeMapper:
    """Returns deterministic matches without loading any model."""

    def find_top_risks(self, text: str, n: int = 3):
        pool = [
            (RiskEntry("Domain A", "Sub A", "definition a"), 0.523456),
            (RiskEntry("Domain B", "Sub B", "definition b"), 0.412345),
            (RiskEntry("Domain C", "Sub C", "definition c"), 0.301234),
        ]
        return pool[:n]


def test_missing_column_raises_value_error():
    df = pd.DataFrame({"text": ["something"]})
    with pytest.raises(ValueError):
        process_descriptions(df, FakeMapper(), text_column="description")


def test_flattens_matches_into_ranked_columns():
    df = pd.DataFrame({"description": ["an ai hiring tool that ranks resumes"]})
    report = process_descriptions(df, FakeMapper(), text_column="description", top_n=2)

    row = report.iloc[0]
    assert row["risk_1_domain"] == "Domain A"
    assert row["risk_1_subdomain"] == "Sub A"
    assert row["risk_2_domain"] == "Domain B"
    # Score is rounded to four decimals.
    assert row["risk_1_score"] == 0.5235
    # top_n=2 means no third rank column.
    assert "risk_3_domain" not in report.columns


def test_empty_rows_get_a_note_and_no_matches():
    df = pd.DataFrame({"description": ["", None, "a real description"]})
    report = process_descriptions(df, FakeMapper(), text_column="description", top_n=1)

    assert report.iloc[0]["note"] == "empty description, skipped"
    assert report.iloc[1]["note"] == "empty description, skipped"
    # The real row has a match and no note value.
    assert report.iloc[2]["risk_1_domain"] == "Domain A"


def test_to_csv_bytes_round_trips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    raw = to_csv_bytes(df)
    assert isinstance(raw, bytes)
    text = raw.decode("utf-8")
    assert "a,b" in text
    assert "1,x" in text
