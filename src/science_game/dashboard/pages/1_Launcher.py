"""Launcher page — start a new evolution run.

Spawns `uv run science-game run ...` as a subprocess so the dashboard
doesn't block. The CLI writes events.jsonl which the Live page tails.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import streamlit as st

from science_game.benchmarks import list_benchmarks
from science_game.dashboard.runs_io import DEFAULT_RUNS_ROOT

st.set_page_config(page_title="Launcher", page_icon=":rocket:", layout="wide")
st.title(":rocket: Launcher")

with st.form("launch"):
    col1, col2, col3 = st.columns(3)
    with col1:
        benchmark = st.selectbox("Benchmark", list_benchmarks(), index=0)
    with col2:
        provider = st.selectbox(
            "LLM provider",
            ["ollama-qwen", "anthropic", "openai"],
            index=0,
            help="anthropic / openai require API keys in env (Sprint 3 adapter pending).",
        )
    with col3:
        model = st.text_input("Model override", value="", help="Leave empty for provider default.")

    col4, col5, col6 = st.columns(3)
    with col4:
        generations = st.number_input("Generations", min_value=1, max_value=10000, value=20)
    with col5:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.1)
    with col6:
        seed = st.number_input("Seed", value=42)

    runs_root = st.text_input("runs/ root", value=str(DEFAULT_RUNS_ROOT))

    submitted = st.form_submit_button("Launch", type="primary")

if submitted:
    cmd = [
        "uv", "run", "science-game", "run", benchmark,
        "--provider", provider,
        "--generations", str(int(generations)),
        "--temperature", str(float(temperature)),
        "--seed", str(int(seed)),
        "--runs-root", runs_root,
    ]
    if model.strip():
        cmd.extend(["--model", model.strip()])

    Path(runs_root).mkdir(parents=True, exist_ok=True)
    log_path = Path(runs_root) / ".launcher.log"

    st.success("Launched.")
    st.code(" ".join(shlex.quote(c) for c in cmd), language="bash")

    with log_path.open("a") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
    st.session_state["last_pid"] = proc.pid
    st.session_state["last_runs_root"] = runs_root
    st.write(
        f"Started subprocess **PID {proc.pid}**. "
        f"Open the **Live** page to watch progress. Subprocess log: `{log_path}`."
    )
