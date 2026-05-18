"""Streamlit dashboard entry point.

Run with:
    uv run streamlit run src/science_game/dashboard/app.py

The dashboard has 4 pages, defined as separate files under dashboard/pages/
(Streamlit auto-discovers them via the `pages/` convention). This entry
page is the Home/Overview.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from science_game.dashboard.runs_io import DEFAULT_RUNS_ROOT, list_runs

st.set_page_config(
    page_title="Algorithm Forge",
    page_icon=":alembic:",
    layout="wide",
)

st.title(":alembic: Algorithm Forge")
st.caption("LLM-driven evolutionary search — research dashboard")

st.markdown(
    """
    **Welcome.** Use the sidebar to navigate:

    - **Launcher** — start a new evolution run with a chosen benchmark + LLM provider.
    - **Live** — watch fitness climb in real time while a run is in progress.
    - **Artifacts** — browse completed runs: best-of-generation code, manifest, mutation log.
    - **Compare** — line up multiple runs on the same plot for ablations.

    The full system design lives in the project plan. Sprint 2 ships this dashboard;
    Phase 4 will turn the per-run mutation logs into a fine-tuning dataset for a
    specialized mutator LLM (via Unsloth on free Colab T4).
    """
)

runs_root = Path(st.sidebar.text_input("runs/ root", value=str(DEFAULT_RUNS_ROOT)))
runs = list_runs(runs_root)

st.subheader(f"Runs in `{runs_root}/`")

if not runs:
    st.info(
        "No runs found yet. Use the **Launcher** page (or "
        "`./scripts/run_local.sh matmul ollama-qwen 5` from the terminal) "
        "to start your first run."
    )
else:
    import pandas as pd

    df = pd.DataFrame([
        {
            "run_id": r.run_id,
            "benchmark": r.benchmark,
            "provider": r.provider,
            "best_fitness": r.best_fitness,
            "generations": r.generations_logged,
            "created_at": r.manifest.get("created_at"),
            "git_sha": (r.manifest.get("git_sha") or "")[:8],
            "dirty": r.manifest.get("git_dirty", False),
        }
        for r in runs
    ])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"{len(runs)} run(s)")
