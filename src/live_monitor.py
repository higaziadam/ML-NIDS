"""Poll a directory for completed CICFlowMeter CSV exports and score them.

The monitor is intentionally flow-file based. It does not capture packets or
read network interfaces; a compatible flow exporter must write CSV files to the
configured incoming directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Callable

from src.flow_ingestion import score_cicflowmeter_csv


logger = logging.getLogger(__name__)
ScoreFunction = Callable[..., dict[str, object]]


def _unique_destination(directory: Path, name: str) -> Path:
    """Return a destination that does not replace a previously handled file."""
    candidate = directory / name
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{Path(name).stem}_{suffix}{Path(name).suffix}"
        suffix += 1
    return candidate


class FlowDirectoryMonitor:
    """Processes stable CSV exports exactly once per observed file path."""

    def __init__(
        self,
        input_dir: str | Path,
        alerts_dir: str | Path,
        processed_dir: str | Path,
        failed_dir: str | Path,
        *,
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        stable_checks: int = 2,
        batch_size: int = 100,
        timeout: float = 30.0,
        score_function: ScoreFunction = score_cicflowmeter_csv,
    ) -> None:
        if stable_checks < 2:
            raise ValueError("stable_checks must be at least 2 to avoid reading an actively written file")
        self.input_dir = Path(input_dir)
        self.alerts_dir = Path(alerts_dir)
        self.processed_dir = Path(processed_dir)
        self.failed_dir = Path(failed_dir)
        self.api_url = api_url
        self.api_key = api_key
        self.stable_checks = stable_checks
        self.batch_size = batch_size
        self.timeout = timeout
        self.score_function = score_function
        self._observations: dict[Path, tuple[int, int, int]] = {}
        for directory in (self.input_dir, self.alerts_dir, self.processed_dir, self.failed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _is_stable(self, path: Path) -> bool:
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        previous = self._observations.get(path)
        if previous is None or previous[:2] != current:
            self._observations[path] = (*current, 1)
            return False
        checks = previous[2] + 1
        self._observations[path] = (*current, checks)
        return checks >= self.stable_checks

    def _move_to(self, source: Path, directory: Path) -> Path:
        destination = _unique_destination(directory, source.name)
        shutil.move(str(source), str(destination))
        return destination

    def _record_failure(self, source: Path, error: Exception) -> Path:
        failed_path = self._move_to(source, self.failed_dir)
        error_path = failed_path.with_suffix(failed_path.suffix + ".error.json")
        with error_path.open("w", encoding="utf-8") as handle:
            json.dump({"source_file": str(failed_path), "error": str(error)}, handle, indent=2)
            handle.write("\n")
        return failed_path

    def _process_file(self, source: Path) -> dict[str, object]:
        # Exporters can reuse a filename after a file is archived. Keep each
        # scored output rather than letting a later export overwrite evidence.
        output = _unique_destination(self.alerts_dir, f"{source.stem}_scored.csv")
        try:
            manifest = self.score_function(
                source,
                output,
                self.api_url,
                api_key=self.api_key,
                batch_size=self.batch_size,
                timeout=self.timeout,
            )
        except Exception as exc:
            failed = self._record_failure(source, exc)
            self._observations.pop(source, None)
            logger.exception("Failed to process flow export %s", source)
            return {"status": "failed", "source_file": str(failed), "error": str(exc)}

        try:
            archived = self._move_to(source, self.processed_dir)
        except Exception as exc:
            # The score already exists, so do not let the watcher retry and
            # duplicate it. Quarantine the source with an explicit status.
            failed = self._record_failure(source, exc)
            self._observations.pop(source, None)
            logger.exception("Scored flow export but could not archive %s", source)
            return {
                "status": "scored_archive_failed",
                "source_file": str(failed),
                "output_path": str(output),
                "manifest": manifest,
                "error": str(exc),
            }

        self._observations.pop(source, None)
        return {"status": "processed", "source_file": str(archived), "manifest": manifest}

    def scan_once(self) -> list[dict[str, object]]:
        """Observe incoming CSV files and process only stable completed files."""
        results: list[dict[str, object]] = []
        candidates = sorted(path for path in self.input_dir.glob("*.csv") if path.is_file())
        active = set(candidates)
        self._observations = {path: observation for path, observation in self._observations.items() if path in active}
        for path in candidates:
            if self._is_stable(path):
                results.append(self._process_file(path))
        return results

    def run_forever(self, poll_interval: float = 5.0) -> None:
        """Poll until interrupted; each loop leaves an audit log through stdout."""
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        logger.info("Watching %s for stable CICFlowMeter CSV exports", self.input_dir)
        while True:
            for result in self.scan_once():
                logger.info("Flow monitor result: %s", json.dumps(result, default=str))
            time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch CICFlowMeter CSV exports and score completed files")
    parser.add_argument("--input-dir", default="runtime/incoming_flows")
    parser.add_argument("--alerts-dir", default="runtime/alerts")
    parser.add_argument("--processed-dir", default="runtime/processed_flows")
    parser.add_argument("--failed-dir", default="runtime/failed_flows")
    parser.add_argument("--api-url", default=os.getenv("ML_NIDS_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("ML_NIDS_API_KEY"))
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    monitor = FlowDirectoryMonitor(
        args.input_dir,
        args.alerts_dir,
        args.processed_dir,
        args.failed_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        stable_checks=args.stable_checks,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    try:
        monitor.run_forever(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Flow monitor stopped")


if __name__ == "__main__":
    main()
