"""Methodological evaluation engine for the MIT AI Risk Mapper.

This module measures how well the vector mapper recovers the correct MIT AI Risk
domain for a curated set of well documented, real world AI incidents. It treats
the task as a single label, multiclass classification problem: for each incident
the mapper predicts the most similar domain, and that prediction is compared
against a human assigned gold label.

We report standard classification metrics computed with scikit-learn:

    Precision: of the incidents the mapper assigned to a domain, how many truly
        belonged to that domain.
    Recall: of the incidents that truly belonged to a domain, how many the
        mapper correctly recovered.
    F1 score: the harmonic mean of precision and recall, which balances the two.

We report both the macro average (every domain weighted equally, which is the
honest view on a balanced dataset) and the weighted average (weighted by class
support). As a secondary retrieval metric we also report top k accuracy, the
fraction of incidents whose correct domain appears among the top k matches.

The vector layer remains the object under test. This script does not change it:
it only quantifies its behaviour so the results are citable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from nlp_mapper import RiskMapper, load_taxonomy

# Default location of the curated gold standard dataset.
DATA_DIR = Path(__file__).resolve().parent / "data"
GOLD_PATH = DATA_DIR / "gold_standard.csv"

# Columns the gold standard file is required to contain.
DESCRIPTION_COLUMN = "description"
LABEL_COLUMN = "expected_domain"


@dataclass
class EvaluationResult:
    """Container for the numbers produced by a single evaluation run.

    Attributes:
        accuracy: Top 1 accuracy, the fraction of exactly correct predictions.
        top_k_accuracy: Fraction of incidents whose true domain is in the top k.
        top_k: The value of k used for the top k accuracy figure.
        macro_precision, macro_recall, macro_f1: Class averaged metrics where
            every domain counts equally.
        weighted_precision, weighted_recall, weighted_f1: Metrics averaged with
            weights proportional to the number of examples in each domain.
        per_class_report: The full scikit-learn report as a nested dictionary,
            keyed by domain, for inspection of individual class performance.
        sample_size: The number of incidents that were evaluated.
    """

    accuracy: float
    top_k_accuracy: float
    top_k: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_report: Dict
    sample_size: int


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, float]:
    """Compute macro and weighted precision, recall, and F1 with scikit-learn.

    This function is deliberately pure: it depends only on the true and predicted
    label sequences, not on the embedding model. That keeps it fast and trivially
    unit testable without loading any heavy resources.

    Args:
        y_true: The gold standard domain labels.
        y_pred: The domain labels predicted by the mapper.
        labels: The full set of valid domain labels, so that domains which never
            appear in a given split are still represented in the averages.

    Returns:
        A dictionary of scalar metrics.
    """
    # zero_division=0 keeps the metric defined (as 0.0) for any domain the mapper
    # never predicts, rather than emitting a runtime warning or a NaN.
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
    }


class MapperEvaluator:
    """Runs the vector mapper over a gold standard set and scores the results.

    The evaluator owns three responsibilities:
        1. Load and validate the gold standard dataset.
        2. Ask the mapper to predict a domain for every incident description.
        3. Compute and present classification metrics for those predictions.
    """

    def __init__(self, mapper: RiskMapper, gold_path: Path = GOLD_PATH) -> None:
        """Store the mapper under test and the path to the gold dataset.

        Args:
            mapper: A ready to use RiskMapper. Injecting it (rather than building
                it inside the evaluator) lets tests pass a lightweight stand in.
            gold_path: Path to the curated gold standard CSV file.
        """
        self.mapper = mapper
        self.gold_path = gold_path
        # The canonical list of domain labels comes from the taxonomy itself, so
        # the metrics always cover every domain even if the sample omits some.
        self.labels: List[str] = [entry.domain for entry in load_taxonomy()]

    def load_gold(self) -> pd.DataFrame:
        """Load the gold standard CSV and validate its shape and labels.

        Returns:
            A DataFrame with the description and expected_domain columns.

        Raises:
            FileNotFoundError: If the gold standard file is missing.
            ValueError: If required columns are absent or a label is unknown.
        """
        if not self.gold_path.exists():
            raise FileNotFoundError(f"Gold standard file not found: {self.gold_path}")

        df = pd.read_csv(self.gold_path)

        for column in (DESCRIPTION_COLUMN, LABEL_COLUMN):
            if column not in df.columns:
                raise ValueError(
                    f"Gold standard is missing the '{column}' column. "
                    f"Found columns: {list(df.columns)}"
                )

        # Guard against typos in the labels: every gold label must be a real
        # taxonomy domain, otherwise the metrics would be meaningless.
        unknown = set(df[LABEL_COLUMN]) - set(self.labels)
        if unknown:
            raise ValueError(
                f"Gold standard contains labels not present in the taxonomy: {unknown}"
            )

        return df

    def _predict_domains(self, description: str, top_k: int) -> List[str]:
        """Return the predicted domains for one description, best match first.

        Args:
            description: The incident text to classify.
            top_k: How many ranked domains to return.

        Returns:
            A list of domain names ordered from most to least similar.
        """
        matches = self.mapper.find_top_risks(description, n=top_k)
        return [entry.domain for entry, _score in matches]

    def run(self, top_k: int = 3) -> EvaluationResult:
        """Evaluate the mapper over the whole gold standard dataset.

        Args:
            top_k: The k used for the secondary top k accuracy metric.

        Returns:
            An EvaluationResult holding every computed figure.
        """
        df = self.load_gold()

        y_true: List[str] = []
        y_pred: List[str] = []
        top_k_hits = 0

        for _, row in df.iterrows():
            description = str(row[DESCRIPTION_COLUMN])
            gold_domain = str(row[LABEL_COLUMN])

            ranked = self._predict_domains(description, top_k=top_k)
            predicted_domain = ranked[0]

            y_true.append(gold_domain)
            y_pred.append(predicted_domain)
            if gold_domain in ranked:
                top_k_hits += 1

        metrics = compute_classification_metrics(y_true, y_pred, self.labels)

        # The full per class breakdown is useful for spotting which domains the
        # mapper confuses. output_dict gives a structured form we can store.
        report = classification_report(
            y_true,
            y_pred,
            labels=self.labels,
            zero_division=0,
            output_dict=True,
        )

        sample_size = len(y_true)
        return EvaluationResult(
            accuracy=metrics["accuracy"],
            top_k_accuracy=top_k_hits / sample_size if sample_size else 0.0,
            top_k=top_k,
            macro_precision=metrics["macro_precision"],
            macro_recall=metrics["macro_recall"],
            macro_f1=metrics["macro_f1"],
            weighted_precision=metrics["weighted_precision"],
            weighted_recall=metrics["weighted_recall"],
            weighted_f1=metrics["weighted_f1"],
            per_class_report=report,
            sample_size=sample_size,
        )

    def print_report(self, result: EvaluationResult) -> None:
        """Print a human readable summary of an evaluation result."""
        print("=" * 70)
        print("MIT AI Risk Mapper: methodological evaluation")
        print("=" * 70)
        print(f"Incidents evaluated : {result.sample_size}")
        print(f"Top 1 accuracy      : {result.accuracy:.3f}")
        print(f"Top {result.top_k} accuracy      : {result.top_k_accuracy:.3f}")
        print()
        print("Macro averaged (every domain weighted equally):")
        print(f"  Precision : {result.macro_precision:.3f}")
        print(f"  Recall    : {result.macro_recall:.3f}")
        print(f"  F1 score  : {result.macro_f1:.3f}")
        print()
        print("Weighted averaged (weighted by class support):")
        print(f"  Precision : {result.weighted_precision:.3f}")
        print(f"  Recall    : {result.weighted_recall:.3f}")
        print(f"  F1 score  : {result.weighted_f1:.3f}")
        print()
        print("Per domain F1 score:")
        for domain in self.labels:
            stats = result.per_class_report.get(domain, {})
            f1 = stats.get("f1-score", 0.0)
            support = int(stats.get("support", 0))
            print(f"  {f1:.3f}  (n={support})  {domain}")
        print("=" * 70)


def main() -> None:
    """Load the mapper, run the evaluation, and print the report."""
    print("Loading taxonomy and embedding model. This can take a moment...")
    mapper = RiskMapper()
    evaluator = MapperEvaluator(mapper)
    result = evaluator.run(top_k=3)
    evaluator.print_report(result)


if __name__ == "__main__":
    main()
