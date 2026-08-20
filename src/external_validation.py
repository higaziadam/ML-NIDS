"""Schema preflight checks for independent external NIDS evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.predict import load_trained_model
from src.utils import save_data


# CIC-IDS2017 CSV exports use older CICFlowMeter header names than the
# CICIDS2018 exports used to train this project.
CICIDS2017_COLUMN_ALIASES = {
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "min_seg_size_forward": "Fwd Seg Size Min",
    "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes",
    "Max Packet Length": "Packet Length Max",
    "Min Packet Length": "Packet Length Min",
}


def normalized_columns(columns: Iterable[object]) -> list[str]:
    """Strip export whitespace and normalize known CIC-IDS2017 aliases."""
    return [CICIDS2017_COLUMN_ALIASES.get(str(column).strip(), str(column).strip()) for column in columns]


def inspect_external_csv(path: str | Path, required_features: Iterable[str]) -> dict[str, object]:
    """Inspect only the header of an external CSV without evaluating a model."""
    csv_path = Path(path)
    columns = normalized_columns(pd.read_csv(csv_path, nrows=0).columns)
    required = list(required_features)
    missing = sorted(set(required) - set(columns))
    duplicate_columns = sorted({column for column in columns if columns.count(column) > 1})
    label_present = "label" in {column.lower() for column in columns}
    return {
        "file": str(csv_path),
        "columns_found": len(columns),
        "label_column_present": label_present,
        "missing_required_features": "; ".join(missing),
        "missing_feature_count": len(missing),
        "duplicate_normalized_columns": "; ".join(duplicate_columns),
        "schema_compatible": bool(label_present and not missing and not duplicate_columns),
    }


def preflight_external_directory(
    data_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Write a header-only compatibility report for independent CSV files."""
    directory = Path(data_dir)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    artifact = load_trained_model(model_path)
    if not isinstance(artifact, dict) or not artifact.get("feature_names"):
        raise ValueError("External preflight requires an artifact with saved feature_names.")

    report = pd.DataFrame(
        [inspect_external_csv(path, artifact["feature_names"]) for path in files]
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_data(report, output_path / "schema_preflight.csv")
    with (output_path / "schema_preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": str(model_path),
                "required_features": artifact["feature_names"],
                "files_inspected": len(report),
                "all_files_schema_compatible": bool(report["schema_compatible"].all()),
                "warning": (
                    "This report checks headers only and does not evaluate model performance. "
                    "Do not substitute missing required features when performing final evaluation."
                ),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight external NIDS CSV schema compatibility")
    parser.add_argument("--data-dir", required=True, help="Directory containing external CSV files")
    parser.add_argument("--model", required=True, help="Saved model artifact with feature schema")
    parser.add_argument("--output", required=True, help="Directory for preflight reports")
    args = parser.parse_args()
    report = preflight_external_directory(args.data_dir, args.model, args.output)
    compatible = int(report["schema_compatible"].sum())
    print(f"Schema-compatible files: {compatible}/{len(report)}")


if __name__ == "__main__":
    main()
