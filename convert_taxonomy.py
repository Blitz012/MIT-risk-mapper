from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def read_taxonomy_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read the taxonomy CSV with sensible defaults and robust encoding handling.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Try UTF-8 first, then fall back to latin-1 if there are decoding issues.
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    except TypeError:
        # Older pandas versions may not support encoding_errors; fall back without it.
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1", encoding_errors="replace")

    return df


def build_taxonomy_structure(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert the flat DataFrame into a nested dictionary structure:

    {
        "Risk Domain": [
            {
                "subdomain": "Risk Subdomain",
                "definition": "Risk Subdomain Definition"
            },
            ...
        ],
        ...
    }
    """
    required_cols = [
        "Risk Domain",
        "Risk Subdomain",
        "Risk Subdomain Definition",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Drop rows where the key fields are all missing or empty
    cleaned = (
        df[required_cols]
        .dropna(how="all")
        .copy()
    )

    # Normalize strings: strip whitespace and drop rows where domain or subdomain is empty after stripping.
    for col in required_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned[
        (cleaned["Risk Domain"] != "")
        & (cleaned["Risk Subdomain"] != "")
    ]

    taxonomy: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in cleaned.iterrows():
        domain = row["Risk Domain"]
        subdomain = row["Risk Subdomain"]
        definition = row["Risk Subdomain Definition"]

        taxonomy.setdefault(domain, []).append(
            {
                "subdomain": subdomain,
                "definition": definition,
            }
        )

    return taxonomy


def main() -> None:
    project_root = Path(__file__).resolve().parent

    csv_filename = "Copy of The AI Risk Repository V4_03_12_2025.xlsx - Domain Taxonomy of AI Risks v1.csv"
    csv_path = project_root / csv_filename

    df = read_taxonomy_csv(csv_path)
    taxonomy = build_taxonomy_structure(df)

    # Ensure output directory exists
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mit_taxonomy.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    print(f"Written taxonomy JSON to: {output_path}")


if __name__ == "__main__":
    main()

