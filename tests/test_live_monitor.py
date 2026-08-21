"""Tests for stable-file processing by the CICFlowMeter directory monitor."""

import json

import pandas as pd

from src.live_monitor import FlowDirectoryMonitor


def _write_source(path) -> None:
    pd.DataFrame({"flow": [1]}).to_csv(path, index=False)


def test_processes_a_file_only_after_two_unchanged_scans(tmp_path) -> None:
    incoming = tmp_path / "incoming"
    source = incoming / "flows.csv"
    incoming.mkdir()
    _write_source(source)

    def score(source_path, output_path, api_url, **kwargs):
        pd.DataFrame({"source_row": [0], "prediction": [1]}).to_csv(output_path, index=False)
        return {"rows_scored": 1, "alerts": 1}

    monitor = FlowDirectoryMonitor(
        incoming,
        tmp_path / "alerts",
        tmp_path / "processed",
        tmp_path / "failed",
        score_function=score,
    )

    assert monitor.scan_once() == []
    results = monitor.scan_once()

    assert results[0]["status"] == "processed"
    assert not source.exists()
    assert (tmp_path / "processed" / "flows.csv").is_file()
    assert (tmp_path / "alerts" / "flows_scored.csv").is_file()


def test_moves_failed_files_and_writes_error_record(tmp_path) -> None:
    incoming = tmp_path / "incoming"
    source = incoming / "bad.csv"
    incoming.mkdir()
    _write_source(source)

    def fail(*args, **kwargs):
        raise RuntimeError("API unavailable")

    monitor = FlowDirectoryMonitor(
        incoming,
        tmp_path / "alerts",
        tmp_path / "processed",
        tmp_path / "failed",
        score_function=fail,
    )
    monitor.scan_once()
    results = monitor.scan_once()

    assert results[0]["status"] == "failed"
    failed = tmp_path / "failed" / "bad.csv"
    assert failed.is_file()
    assert json.loads(failed.with_suffix(".csv.error.json").read_text(encoding="utf-8"))["error"] == "API unavailable"


def test_quarantines_source_when_scoring_succeeds_but_archiving_fails(tmp_path, monkeypatch) -> None:
    incoming = tmp_path / "incoming"
    source = incoming / "scored.csv"
    incoming.mkdir()
    _write_source(source)

    def score(source_path, output_path, api_url, **kwargs):
        pd.DataFrame({"source_row": [0], "prediction": [0]}).to_csv(output_path, index=False)
        return {"rows_scored": 1, "alerts": 0}

    monitor = FlowDirectoryMonitor(
        incoming,
        tmp_path / "alerts",
        tmp_path / "processed",
        tmp_path / "failed",
        score_function=score,
    )
    original_move = monitor._move_to

    def fail_processed_move(source_path, directory):
        if directory == monitor.processed_dir:
            raise OSError("archive unavailable")
        return original_move(source_path, directory)

    monkeypatch.setattr(monitor, "_move_to", fail_processed_move)
    monitor.scan_once()
    results = monitor.scan_once()

    assert results[0]["status"] == "scored_archive_failed"
    assert (tmp_path / "alerts" / "scored_scored.csv").is_file()
    assert (tmp_path / "failed" / "scored.csv").is_file()
