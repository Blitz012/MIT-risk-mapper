"""Bulk CSV processing for the MIT AI Risk Mapping Tool.

Takes a DataFrame of project descriptions and runs each one through a single
shared RiskMapper, returning a flattened report of the top vector matches and
their cosine scores. Reusing one mapper across all rows is what keeps bulk runs
fast: the embedding model and taxonomy are loaded once, not once per row.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from nlp_mapper import RiskMapper


def process_descriptions(
    df: pd.DataFrame,
    mapper: RiskMapper,
    text_column: str,
    top_n: int = 3,
) -> pd.DataFrame:
    """Audit every row of df and return a flattened report DataFrame.

    For each description the report keeps the original text plus, for each rank
    up to top_n, the matched domain, subdomain, and cosine score. Rows with an
    empty or missing description are reported with a note and no matches.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    records = []
    for _, row in df.iterrows():
        value = row[text_column]
        text = "" if pd.isna(value) else str(value).strip()

        record = {text_column: value}
        if not text:
            record["note"] = "empty description, skipped"
            records.append(record)
            continue

        matches = mapper.find_top_risks(text, n=top_n)
        for rank, (entry, score) in enumerate(matches, start=1):
            record[f"risk_{rank}_domain"] = entry.domain
            record[f"risk_{rank}_subdomain"] = entry.subdomain
            record[f"risk_{rank}_score"] = round(float(score), 4)
        records.append(record)

    return pd.DataFrame(records)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 CSV bytes for a Streamlit download button."""
    return df.to_csv(index=False).encode("utf-8")


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "description": [
                "A health app that predicts patient outcomes from medical records.",
                "An AI hiring tool that ranks resumes automatically.",
                "",
            ]
        }
    )
    print("Loading mapper...")
    test_mapper = RiskMapper()
    report = process_descriptions(sample, test_mapper, text_column="description", top_n=3)
    print(report.to_string(index=False))
