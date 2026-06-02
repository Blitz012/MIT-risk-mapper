from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util


DATA_DIR = Path(__file__).resolve().parent / "data"
TAXONOMY_PATH = DATA_DIR / "mit_taxonomy.json"
MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RiskEntry:
    domain: str
    subdomain: str
    definition: str


def load_taxonomy(path: Path = TAXONOMY_PATH) -> List[RiskEntry]:
    """
    Load the MIT taxonomy JSON and flatten it into a list of RiskEntry objects.
    """
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries: List[RiskEntry] = []
    for domain, items in data.items():
        for item in items:
            sub = item.get("subdomain", "").strip()
            definition = item.get("definition", "").strip()
            if not definition:
                continue
            entries.append(RiskEntry(domain=domain, subdomain=sub, definition=definition))

    if not entries:
        raise ValueError("No valid risk entries found in taxonomy.")

    return entries


class RiskMapper:
    """
    Encapsulates the embedding model and pre-computed taxonomy vectors.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.entries: List[RiskEntry] = load_taxonomy()

        # Pre-compute embeddings for all definitions
        definitions = [e.definition for e in self.entries]
        self.definition_embeddings = self.model.encode(definitions, convert_to_tensor=True, normalize_embeddings=True)

    def find_top_risks(self, user_input: str, n: int = 3) -> List[Tuple[RiskEntry, float]]:
        """
        Compute cosine similarity between the user_input and all risk definitions.

        Returns a list of (RiskEntry, score) sorted by descending score.
        """
        if not user_input or not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")

        # Embed user input
        query_embedding = self.model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)

        # Cosine similarity with all definition embeddings
        cosine_scores = util.cos_sim(query_embedding, self.definition_embeddings)[0]  # shape: [num_entries]

        # Get top-n indices
        n = min(n, len(self.entries))
        top_indices = np.argsort(-cosine_scores.cpu().numpy())[:n]

        results: List[Tuple[RiskEntry, float]] = []
        for idx in top_indices:
            entry = self.entries[int(idx)]
            score = float(cosine_scores[int(idx)].item())
            results.append((entry, score))

        return results


def find_top_risks(user_input: str, n: int = 3):
    """
    Convenience function that instantiates a RiskMapper and performs a lookup.

    Returns a list of dictionaries:
    [
      {"domain": ..., "subdomain": ..., "definition": ..., "score": ...},
      ...
    ]
    """
    mapper = RiskMapper()
    results = mapper.find_top_risks(user_input, n=n)
    return [
        {
            "domain": entry.domain,
            "subdomain": entry.subdomain,
            "definition": entry.definition,
            "score": score,
        }
        for entry, score in results
    ]


if __name__ == "__main__":
    sample_text = "A health app that uses AI to predict patient outcomes based on historical records"
    print("Loading taxonomy and embedding model...")
    mapper = RiskMapper()
    print(f"Running test query: {sample_text!r}\n")
    top_results = mapper.find_top_risks(sample_text, n=3)

    for i, (entry, score) in enumerate(top_results, start=1):
        print(f"Match {i}:")
        print(f"  Domain    : {entry.domain}")
        print(f"  Subdomain : {entry.subdomain}")
        print(f"  Score     : {score:.4f}")
        print()

