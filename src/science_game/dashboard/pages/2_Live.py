"""Live view — tail events.jsonl of a chosen run, plot fitness, show last mutation."""

from __future__ import annotations

import time
from pathlib import Path

import plotly.express as px
import streamlit as st

from science_game.dashboard.runs_io import (
    DEFAULT_RUNS_ROOT,
    list_best_files,
    list_runs,
    load_events,
    load_mutations,
)

st.set_page_config(page_title="Live", page_icon=":satellite:", layout="wide")
st.title(":satellite: Live")

runs_root = Path(st.sidebar.text_input("runs/ root", value=str(DEFAULT_RUNS_ROOT)))
runs = list_runs(runs_root)

if not runs:
    st.info("No runs yet. Start one from the **Launcher** page.")
    st.stop()

run_ids = [r.run_id for r in runs]
selected = st.sidebar.selectbox("Run", run_ids, index=0)
run = next(r for r in runs if r.run_id == selected)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh every (s)", 1, 30, 3)

df = load_events(run.path)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Generations", int(df["generation"].max()) if not df.empty else 0)
m2.metric("Best fitness", f"{run.best_fitness:.6f}" if run.best_fitness is not None else "—")
m3.metric("Benchmark", run.benchmark)
m4.metric("Provider", run.provider)

if df.empty:
    st.info("No events yet — run probably starting up.")
else:
    fig = px.line(
        df, x="generation", y=["fitness", "best_so_far"] if "best_so_far" in df else ["fitness"],
        markers=True, title="Fitness trajectory",
    )
    fig.update_layout(legend_title_text="series", height=420)
    st.plotly_chart(fig, width="stretch")

mut_df = load_mutations(run.path)
if not mut_df.empty:
    st.subheader("Latest mutation")
    last = mut_df.iloc[-1]
    c1, c2 = st.columns(2)
    c1.markdown("**Parent code**")
    c1.code(last.get("parent_code", ""), language="python")
    c2.markdown(
        f"**Child code** — accepted: {bool(last.get('accepted'))} | "
        f"fitness: {last.get('fitness_before'):.6f} → {last.get('fitness_after'):.6f}"
    )
    c2.code(last.get("child_code", ""), language="python")

best_files = list_best_files(run.path)
if best_files:
    st.subheader(f"Best-of-generation snapshots ({len(best_files)})")
    chosen = st.selectbox(
        "Snapshot",
        best_files,
        format_func=lambda p: p.name,
        index=len(best_files) - 1,
    )
    st.code(Path(chosen).read_text(encoding="utf-8"), language="python")

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
