from __future__ import annotations

import pandas as pd
import streamlit as st

from bulk_processor import process_descriptions, to_csv_bytes
from llm_evaluator import RationaleError, generate_rationale
from nlp_mapper import RiskMapper
from report_builder import (
    build_analysis_dataframe,
    build_markdown_report,
    report_to_bytes,
)
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


def _render_analysis(analysis: dict) -> None:
    """Render a stored analysis: match cards, radar, rationale, and downloads."""
    user_input = analysis["input"]
    results = analysis["results"]
    rationale = analysis["rationale"]
    rationale_error = analysis["rationale_error"]

    st.subheader("Top Matching Risks")

    # Normalize scores to [0,1] for the progress bars. Cosine scores can sit in
    # a narrow band, so this stretches them relative to the current matches.
    raw_scores = [r["score"] for r in results]
    min_score = min(raw_scores) if raw_scores else 0.0
    max_score = max(raw_scores) if raw_scores else 0.0

    def normalize(score: float) -> float:
        if max_score == min_score:
            return 1.0
        return (score - min_score) / (max_score - min_score)

    for i, r in enumerate(results, start=1):
        with st.container():
            cols = st.columns([3, 3, 2, 2])
            with cols[0]:
                st.markdown(f"**Match {i}**")
            with cols[1]:
                st.markdown(f"**Domain:** {r['domain']}")
                st.markdown(f"**Subdomain:** {r['subdomain']}")
            with cols[2]:
                st.metric(label="Similarity (cosine)", value=f"{r['score']:.3f}")
            with cols[3]:
                st.progress(normalize(r["score"]))

    st.subheader("Risk Similarity Radar")
    radar_fig = build_radar(results)
    if radar_fig is not None:
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.caption("Not enough matches to draw a radar chart.")

    st.subheader("Why these risks apply")
    if rationale_error:
        st.info(f"Rationale unavailable, showing vector matches only. {rationale_error}")
    if rationale:
        if rationale.get("summary"):
            st.markdown(f"**Summary:** {rationale['summary']}")
        for item in rationale.get("rationales", []):
            sub = item.get("subdomain", "")
            text = item.get("rationale", "")
            with st.expander(sub or "Risk rationale", expanded=True):
                st.write(text)

    st.subheader("Download this analysis")
    csv_bytes = to_csv_bytes(build_analysis_dataframe(results))
    md_bytes = report_to_bytes(build_markdown_report(user_input, results, rationale))
    dl_cols = st.columns(2)
    with dl_cols[0]:
        st.download_button(
            label="Download matches CSV",
            data=csv_bytes,
            file_name="risk_analysis.csv",
            mime="text/csv",
        )
    with dl_cols[1]:
        st.download_button(
            label="Download report (Markdown)",
            data=md_bytes,
            file_name="risk_analysis_report.md",
            mime="text/markdown",
        )


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
            results = get_top_risks(user_input, n=num_matches)

        # Generate the rationale once, here, rather than on every rerun. The
        # whole analysis is cached in session_state so that download clicks
        # (which rerun the script) neither call the model again nor wipe the
        # results off the page.
        rationale = None
        rationale_error = None
        with st.spinner("Generating rationale..."):
            try:
                rationale = generate_rationale(user_input, results)
            except RationaleError as err:
                rationale_error = str(err)

        st.session_state["analysis"] = {
            "input": user_input,
            "results": results,
            "rationale": rationale,
            "rationale_error": rationale_error,
        }

    analysis = st.session_state.get("analysis")
    if analysis:
        _render_analysis(analysis)

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

