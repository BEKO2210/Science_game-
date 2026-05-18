"""Compare multiple runs on one fitness plot — ablation view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from science_game.dashboard.runs_io import DEFAULT_RUNS_ROOT, list_runs, load_events

st.set_page_config(page_title="Compare", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: Compare runs")

runs_root = Path(st.sidebar.text_input("runs/ root", value=str(DEFAULT_RUNS_ROOT)))
runs = list_runs(runs_root)

if not runs:
    st.info("No runs yet.")
    st.stop()

selected = st.multiselect(
    "Runs to compare",
    [r.run_id for r in runs],
    default=[r.run_id for r in runs[: min(3, len(runs))]],
)

if not selected:
    st.info("Select at least one run.")
    st.stop()

frames: list[pd.DataFrame] = []
for run_id in selected:
    run = next(r for r in runs if r.run_id == run_id)
    df = load_events(run.path)
    if df.empty:
        continue
    df = df[["generation", "best_so_far"]].copy()
    df["run"] = run_id
    frames.append(df)

if not frames:
    st.warning("No events in selected runs.")
    st.stop()

combined = pd.concat(frames, ignore_index=True)
fig = px.line(
    combined, x="generation", y="best_so_far", color="run", markers=True,
    title="Best fitness so far",
)
fig.update_layout(height=480)
st.plotly_chart(fig, width="stretch")
