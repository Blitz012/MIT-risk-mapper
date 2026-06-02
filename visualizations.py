"""Interactive visualizations for the MIT AI Risk Mapping Tool.

Builds a Plotly radar chart that maps cosine similarity scores across the top
matching risk subdomains, giving a quick visual sense of how strongly a project
relates to each risk.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go


def _short_label(text: str, max_len: int = 28) -> str:
    """Shorten long subdomain names so the radar axis labels stay readable."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def build_radar(
    matches: List[Dict],
    title: str = "Risk similarity radar",
) -> Optional[go.Figure]:
    """Build a radar (polar) chart from a list of match dicts.

    Each match must have 'subdomain' and 'score'. Returns None if there are
    fewer than three matches, since a radar needs at least three axes to read
    as an area rather than a line.
    """
    if not matches or len(matches) < 3:
        return None

    labels = [_short_label(m["subdomain"]) for m in matches]
    scores = [float(m["score"]) for m in matches]

    # Close the loop so the filled polygon connects back to the first axis.
    labels_closed = labels + [labels[0]]
    scores_closed = scores + [scores[0]]

    upper = max(scores) * 1.1 if max(scores) > 0 else 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=labels_closed,
            fill="toself",
            name="Cosine similarity",
            hovertemplate="%{theta}<br>Similarity: %{r:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, upper])),
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


if __name__ == "__main__":
    demo = [
        {"subdomain": "Unfair discrimination and misrepresentation", "score": 0.28},
        {"subdomain": "Compromise of privacy", "score": 0.25},
        {"subdomain": "Disinformation, surveillance, and influence at scale", "score": 0.24},
    ]
    fig = build_radar(demo)
    print("Figure built:", fig is not None)
    print("Traces:", len(fig.data))
    print("Axis labels:", list(fig.data[0].theta))
