from __future__ import annotations

import pandas as pd
import streamlit as st

from bulk_processor import process_descriptions, to_csv_bytes
from llm_evaluator import RationaleError, generate_rationale
from nlp_mapper import RiskMapper
from visualizations import build_radar


@st.cache_resource(show_spinner="Loading model and taxonomy...")
def get_mapper() -> RiskMapper:
    """Load the embedding model and taxonomy once, then reuse across reruns."""
    return RiskMapper()


def get_top_risks(user_input: str, n: int = 3) -> list[dict]:
    """Run the cached mapper and return result dicts including the definition."""
    mapper = get_mapper()
    return [
        {
            "domain": entry.domain,
            "subdomain": entry.subdomain,
            "definition": entry.definition,
            "score": score,
        }
        for entry, score in mapper.find_top_risks(user_input, n=n)
    ]


def main() -> None:
    st.set_page_config(page_title="MIT AI Risk Mapping Tool", layout="wide")

    st.title("MIT AI Risk Mapping Tool")
    st.write(
        "Paste an AI project description below and click **Analyze Risks** to see the "
        "closest matching risk domains and subdomains from the MIT taxonomy."
    )

    default_text = (
        "A health app that uses AI to predict patient outcomes based on historical records."
    )
    user_input = st.text_area(
        "AI project description",
        value=default_text,
        height=200,
        help="Describe your AI project in as much detail as possible.",
    )

    num_matches = st.slider(
        "Number of top matches to show",
        min_value=3,
        max_value=8,
        value=3,
        help="How many of the closest risk subdomains to display and chart.",
    )

    analyze = st.button("Analyze Risks")

    if analyze:
        stripped = user_input.strip()
        if not stripped:
            st.warning("Please enter an AI project description before analyzing.")
            return
        if len(stripped.split()) < 4:
            st.warning(
                "Please enter a more complete description (at least a short "
                "sentence) so the match is meaningful."
            )
            return

        with st.spinner("Analyzing risks..."):
            # Fetch exactly the number the user asked for. Three or more keeps
            # the radar readable as an area rather than a line.
            radar_results = get_top_risks(user_input, n=num_matches)
            results = radar_results

        st.subheader("Top Matching Risks")

        # Normalize scores to [0,1] for visualization if they are cosine similarities in [-1,1]
        raw_scores = [r["score"] for r in results]
        if raw_scores:
            min_score = min(raw_scores)
            max_score = max(raw_scores)
        else:
            min_score = max_score = 0.0

        def normalize(score: float) -> float:
            if max_score == min_score:
                return 1.0
            return (score - min_score) / (max_score - min_score)

        for i, r in enumerate(results, start=1):
            domain = r["domain"]
            subdomain = r["subdomain"]
            score = r["score"]
            norm_score = normalize(score)

            with st.container():
                cols = st.columns([3, 3, 2, 2])
                with cols[0]:
                    st.markdown(f"**Match {i}**")
                with cols[1]:
                    st.markdown(f"**Domain:** {domain}")
                    st.markdown(f"**Subdomain:** {subdomain}")
                with cols[2]:
                    st.metric(label="Similarity (cosine)", value=f"{score:.3f}")
                with cols[3]:
                    st.progress(norm_score)

        st.subheader("Risk Similarity Radar")
        radar_fig = build_radar(radar_results)
        if radar_fig is not None:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.caption("Not enough matches to draw a radar chart.")

        st.subheader("Why these risks apply")
        with st.spinner("Generating rationale..."):
            try:
                rationale = generate_rationale(user_input, results)
            except RationaleError as err:
                rationale = None
                st.info(
                    f"Rationale unavailable, showing vector matches only. {err}"
                )

        if rationale:
            if rationale.get("summary"):
                st.markdown(f"**Summary:** {rationale['summary']}")
            for item in rationale.get("rationales", []):
                subdomain = item.get("subdomain", "")
                text = item.get("rationale", "")
                with st.expander(subdomain or "Risk rationale", expanded=True):
                    st.write(text)

    st.divider()
    st.subheader("Bulk CSV Audit")
    st.write(
        "Upload a CSV of project descriptions to audit them in bulk and download "
        "a report of the top risk matches and scores for each one."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            bulk_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            return

        if bulk_df.empty:
            st.warning("The uploaded CSV has no rows.")
            return

        st.caption(f"Loaded {len(bulk_df)} rows and {len(bulk_df.columns)} columns.")

        text_column = st.selectbox(
            "Which column holds the project descriptions?",
            options=list(bulk_df.columns),
        )
        bulk_top_n = st.slider(
            "Matches per description",
            min_value=1,
            max_value=5,
            value=3,
            help="How many top risk matches to record for each row.",
        )

        if st.button("Run Bulk Audit"):
            with st.spinner(f"Auditing {len(bulk_df)} descriptions..."):
                mapper = get_mapper()
                report = process_descriptions(
                    bulk_df, mapper, text_column=text_column, top_n=bulk_top_n
                )

            # A scored row has no note; skipped (empty) rows carry a note value.
            if "note" in report.columns:
                scored = int(report["note"].isna().sum())
            else:
                scored = len(report)
            skipped = len(report) - scored
            if skipped:
                st.success(
                    f"Processed {len(report)} rows: {scored} scored, "
                    f"{skipped} skipped as empty."
                )
            else:
                st.success(f"Processed {len(report)} rows.")
            st.dataframe(report, use_container_width=True)
            st.download_button(
                label="Download report CSV",
                data=to_csv_bytes(report),
                file_name="risk_audit_report.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

