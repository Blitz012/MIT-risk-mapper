"""Offline tests for the methodological evaluation engine.

These tests avoid loading the heavy sentence-transformers model. The pure metric
function is exercised directly, and the MapperEvaluator is driven by a lightweight
FakeMapper stand in so the whole suite stays fast and runs without a network.

The only real resource these tests touch is the taxonomy JSON, which the evaluator
reads to learn the canonical domain labels. That read is pure file parsing and is
already covered by the taxonomy tests, so it is safe to rely on here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pytest

from evaluate_mapper import (
    MapperEvaluator,
    compute_classification_metrics,
)
from nlp_mapper import RiskEntry, load_taxonomy


# The canonical domain labels, taken from the taxonomy itself so the tests stay
# in sync with the real data rather than hard coding a possibly stale list.
LABELS = [entry.domain for entry in load_taxonomy()]


class FakeMapper:
    """A minimal stand in for RiskMapper used to drive the evaluator offline.

    Instead of embedding text, it maps each description to a pre-arranged ranked
    list of domains. This lets the tests control exactly what the evaluator sees
    and assert on the resulting metrics without any model.
    """

    def __init__(self, scripted: dict) -> None:
        """Store a mapping from description text to an ordered list of domains.

        Args:
            scripted: Keys are description strings, values are lists of domain
                names ordered from best to worst match.
        """
        self.scripted = scripted

    def find_top_risks(
        self, description: str, n: int = 3
    ) -> List[Tuple[RiskEntry, float]]:
        """Return scripted (RiskEntry, score) pairs for a description.

        The score is synthesised as a simple descending sequence so that the
        ordering, which is all the evaluator cares about, is preserved.
        """
        domains = self.scripted.get(description, [])[:n]
        results: List[Tuple[RiskEntry, float]] = []
        for rank, domain in enumerate(domains):
            entry = RiskEntry(domain=domain, subdomain="", definition="")
            results.append((entry, 1.0 - 0.1 * rank))
        return results


def test_compute_metrics_perfect_prediction():
    """When every prediction is correct, all metrics should equal 1.0."""
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "C"]
    metrics = compute_classification_metrics(y_true, y_pred, labels=["A", "B", "C"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0


def test_compute_metrics_all_wrong():
    """When no prediction is correct, accuracy and F1 should be 0.0."""
    y_true = ["A", "A", "B"]
    y_pred = ["B", "B", "A"]
    metrics = compute_classification_metrics(y_true, y_pred, labels=["A", "B"])
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["weighted_f1"] == 0.0


def test_compute_metrics_handles_unpredicted_label_without_error():
    """A label the mapper never predicts must not raise, just score 0 for it.

    With zero_division=0 the precision and recall for the missing class are
    defined as 0.0 rather than NaN, so the macro average stays a real number.
    """
    y_true = ["A", "B"]
    y_pred = ["A", "A"]
    metrics = compute_classification_metrics(y_true, y_pred, labels=["A", "B", "C"])
    assert metrics["accuracy"] == 0.5
    # Macro F1 averages over A, B, and C; B and C contribute 0, so it is below 1.
    assert 0.0 < metrics["macro_f1"] < 1.0


def _build_evaluator(tmp_path, rows, scripted):
    """Helper that writes a temporary gold CSV and wires up an evaluator.

    Args:
        tmp_path: pytest fixture for an isolated temp directory.
        rows: list of (description, expected_domain) tuples for the gold file.
        scripted: the FakeMapper script mapping description to ranked domains.
    """
    import csv

    gold_path = tmp_path / "gold.csv"
    with gold_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["description", "expected_domain"])
        for description, domain in rows:
            writer.writerow([description, domain])

    mapper = FakeMapper(scripted)
    return MapperEvaluator(mapper, gold_path=gold_path)


def test_run_perfect_scores_when_mapper_always_right(tmp_path):
    """End to end run where the FakeMapper returns the gold domain first.

    Every taxonomy domain appears once in the gold set so that the macro average,
    which spans all labels, can reach 1.0. This mirrors the real balanced run
    where each domain has support.
    """
    rows = [(f"desc {i}", domain) for i, domain in enumerate(LABELS)]
    scripted = {f"desc {i}": [domain] for i, domain in enumerate(LABELS)}
    evaluator = _build_evaluator(tmp_path, rows, scripted)
    result = evaluator.run(top_k=3)
    assert result.sample_size == len(LABELS)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.weighted_f1 == 1.0
    assert result.top_k_accuracy == 1.0


def test_run_top_k_accuracy_rewards_lower_rank_hit(tmp_path):
    """A correct domain at rank 2 should miss top 1 but make top k."""
    d0, d1 = LABELS[0], LABELS[1]
    rows = [("desc zero", d0)]
    # The gold domain d0 is the second match, so top 1 is wrong but top 3 hits.
    scripted = {"desc zero": [d1, d0]}
    evaluator = _build_evaluator(tmp_path, rows, scripted)
    result = evaluator.run(top_k=3)
    assert result.accuracy == 0.0
    assert result.top_k_accuracy == 1.0


def test_load_gold_missing_file_raises(tmp_path):
    """A non existent gold path should raise FileNotFoundError."""
    mapper = FakeMapper({})
    evaluator = MapperEvaluator(mapper, gold_path=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        evaluator.load_gold()


def test_load_gold_missing_column_raises(tmp_path):
    """A gold file lacking a required column should raise ValueError."""
    import csv

    gold_path = tmp_path / "bad.csv"
    with gold_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["description", "wrong_label_column"])
        writer.writerow(["something", "value"])

    mapper = FakeMapper({})
    evaluator = MapperEvaluator(mapper, gold_path=gold_path)
    with pytest.raises(ValueError):
        evaluator.load_gold()


def test_load_gold_unknown_label_raises(tmp_path):
    """A gold label that is not a real taxonomy domain should raise ValueError."""
    rows = [("desc", "This Is Not A Real Domain")]
    evaluator = _build_evaluator(tmp_path, rows, {})
    with pytest.raises(ValueError):
        evaluator.load_gold()
