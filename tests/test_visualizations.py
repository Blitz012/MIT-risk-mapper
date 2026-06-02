"""Tests for the Plotly radar chart builder.

These are pure and fast: they exercise no embedding model and no network.
"""
from __future__ import annotations

from visualizations import _short_label, build_radar


def _matches(n: int):
    return [{"subdomain": f"Risk subdomain {i}", "score": 0.5 - i * 0.05} for i in range(n)]


def test_short_label_keeps_short_text():
    assert _short_label("Compromise of privacy") == "Compromise of privacy"


def test_short_label_truncates_long_text():
    long = "A very long subdomain name that should be truncated for the axis"
    out = _short_label(long, max_len=20)
    assert len(out) <= 20
    assert out.endswith("…")


def test_build_radar_returns_none_for_too_few_matches():
    assert build_radar([]) is None
    assert build_radar(_matches(2)) is None


def test_build_radar_returns_figure_for_three_or_more():
    fig = build_radar(_matches(3))
    assert fig is not None
    # One Scatterpolar trace.
    assert len(fig.data) == 1


def test_build_radar_closes_the_loop():
    fig = build_radar(_matches(4))
    # The polygon is closed, so theta and r have one extra point appended.
    assert len(fig.data[0].theta) == 5
    assert len(fig.data[0].r) == 5
    assert fig.data[0].theta[0] == fig.data[0].theta[-1]
