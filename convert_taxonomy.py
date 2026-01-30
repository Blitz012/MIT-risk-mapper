from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def read_taxonomy_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    Read the taxonomy data from the specified Excel sheet.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    sheet_name = "Domain Taxonomy of AI Risks v1"

    # Some exports include title/metadata rows above the real column headers.
    # Per requirement: scan the first 5 rows first.
    def find_header_row(nrows: int) -> int | None:
        preview = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=nrows)
        for i in range(len(preview.index)):
            row = preview.iloc[i].astype(str)
            # Use a simple case-insensitive substring match; Excel cells sometimes include extra whitespace.
            has_domain = row.str.contains("Domain", case=False, regex=False).any()
            has_subdomain = row.str.contains("Sub-domain", case=False, regex=False).any() or row.str.contains(
                "Subdomain", case=False, regex=False
            ).any()
            if has_domain and has_subdomain:
                return i
        return None

    header_row_idx = find_header_row(5)
    if header_row_idx is None:
        # Fallback for files where the header row is further down.
        header_row_idx = find_header_row(100)
        if header_row_idx is None:
            raise ValueError(
                "Could not find a header row containing 'Risk Domain' in the first 5 rows "
                "(or within the first 100 rows fallback)."
            )

    # Print 1-based row for human readability.
    print(f"Found headers on row {header_row_idx + 1}")

    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        header=header_row_idx,
    )

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
    def norm(col: str) -> str:
        return " ".join(str(col).replace("\n", " ").split()).strip().lower()

    normalized_to_original: Dict[str, str] = {norm(c): c for c in df.columns}

    def pick_col(candidates: list[str]) -> str | None:
        for c in candidates:
            key = norm(c)
            if key in normalized_to_original:
                return normalized_to_original[key]
        return None

    domain_col = pick_col(["Risk Domain", "Domain"])
    subdomain_col = pick_col(["Risk Subdomain", "Sub-domain", "Subdomain"])
    definition_col = pick_col(["Risk Subdomain Definition", "Risk Subdomain Definition ", "Description", "Description "])

    missing_named = [
        name
        for name, col in [
            ("Risk Domain", domain_col),
            ("Risk Subdomain", subdomain_col),
            ("Risk Subdomain Definition", definition_col),
        ]
        if col is None
    ]
    if missing_named:
        raise ValueError(
            f"Missing required columns in data: {missing_named}. "
            f"Detected columns: {list(df.columns)}"
        )

    # Drop rows where the key fields are all missing or empty
    cleaned = (
        df[[domain_col, subdomain_col, definition_col]]
        .dropna(how="all")
        .copy()
    )

    # Normalize strings: strip whitespace and drop rows where domain or subdomain is empty after stripping.
    for col in [domain_col, subdomain_col, definition_col]:
        cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    cleaned = cleaned[
        (cleaned[domain_col] != "")
        & (cleaned[subdomain_col] != "")
    ]

    taxonomy: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in cleaned.iterrows():
        domain = row[domain_col]
        subdomain = row[subdomain_col]
        definition = row[definition_col]

        taxonomy.setdefault(domain, []).append(
            {
                "subdomain": subdomain,
                "definition": definition,
            }
        )

    return taxonomy


def main() -> None:
    project_root = Path(__file__).resolve().parent

    xlsx_filename = "Copy of The AI Risk Repository V4_03_12_2025.xlsx"
    xlsx_path = project_root / xlsx_filename

    print("Loading file...")
    df = read_taxonomy_excel(xlsx_path)

    print("Processing domains...")
    taxonomy = build_taxonomy_structure(df)

    # Ensure output directory exists
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mit_taxonomy.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    print("Success!")
    print(f"Written taxonomy JSON to: {output_path}")


if __name__ == "__main__":
    main()

