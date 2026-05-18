"""Artifacts page — browse a finished run's artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from science_game.dashboard.runs_io import (
    DEFAULT_RUNS_ROOT,
    list_best_files,
    list_runs,
    load_mutations,
)

st.set_page_config(page_title="Artifacts", page_icon=":package:", layout="wide")
st.title(":package: Artifacts")

runs_root = Path(st.sidebar.text_input("runs/ root", value=str(DEFAULT_RUNS_ROOT)))
runs = list_runs(runs_root)

if not runs:
    st.info("No runs yet.")
    st.stop()

selected = st.sidebar.selectbox("Run", [r.run_id for r in runs])
run = next(r for r in runs if r.run_id == selected)

st.subheader(f"Run `{run.run_id}`")

st.markdown("**Manifest**")
st.json(run.manifest, expanded=False)

best_files = list_best_files(run.path)
if best_files:
    st.markdown(f"**Best-of-generation files ({len(best_files)})**")
    latest = run.path / "best" / "latest.py"
    if latest.exists():
        st.markdown("`best/latest.py` (current best):")
        st.code(latest.read_text(), language="python")

mut_df = load_mutations(run.path)
if not mut_df.empty:
    st.markdown(f"**Mutation log** — {len(mut_df)} rows")
    st.dataframe(
        mut_df[[c for c in ["generation", "accepted", "fitness_before", "fitness_after", "provider", "model", "tokens_in", "tokens_out"] if c in mut_df.columns]],
        hide_index=True, width="stretch",
    )

st.markdown("**Files on disk**")
files = [p for p in run.path.rglob("*") if p.is_file()]
st.code("\n".join(str(p.relative_to(run.path)) for p in sorted(files)), language="text")

st.divider()
st.markdown("### Publish (Sprint 3)")
st.caption("Upload best-of-run + manifest to Hugging Face Hub — wired up in the next sprint.")
disabled = True
st.button("Publish to HF Hub", disabled=disabled, help="Sprint 3.")
