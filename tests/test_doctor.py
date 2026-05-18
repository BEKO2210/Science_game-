"""Tests for the doctor preflight checks."""

from __future__ import annotations

from science_game.doctor import (
    check_dashboard_extras,
    check_python_version,
    check_runs_dir,
    check_submodule,
    run_all_checks,
)


def test_python_version_passes():
    r = check_python_version()
    assert r.ok
    assert "." in r.summary


def test_runs_dir_never_fails():
    r = check_runs_dir()
    assert r.ok


def test_dashboard_extras_check_truthful():
    r = check_dashboard_extras()
    # We installed dashboard extras for the test env — should pass.
    assert r.ok, r.hint


def test_submodule_check_works():
    r = check_submodule()
    assert isinstance(r.ok, bool)
    assert r.summary


def test_run_all_returns_list_of_results():
    results = run_all_checks()
    assert len(results) > 5
    assert all(hasattr(r, "ok") and hasattr(r, "name") for r in results)
