"""Schema preflight checks for independent external NIDS evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.predict import load_trained_model
from src.release_config import load_release_profile, sha256_file
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


def _standardize_cicids2017_chunk(
    chunk: pd.DataFrame, required_features: Iterable[str]
) -> tuple[pd.DataFrame, int]:
    """Normalize one CIC-IDS2017 chunk without fitting any data transforms."""
    data = chunk.copy()
    data.columns = normalized_columns(data.columns)
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Normalized external schema has duplicate columns: {duplicates}")
    label_columns = [column for column in data.columns if column.lower() == "label"]
    if len(label_columns) != 1:
        raise ValueError("External CIC-IDS2017 chunk must contain exactly one Label column.")
    required = list(required_features)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"External CIC-IDS2017 chunk is missing required features: {missing}")

    labels = data[label_columns[0]].astype("string").str.strip()
    features = data.loc[:, required].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    valid = labels.notna() & labels.ne("") & features.notna().all(axis=1)
    dropped = int((~valid).sum())
    prepared = features.loc[valid].copy()
    prepared["label"] = labels.loc[valid].ne("BENIGN").astype(int).to_numpy()
    return prepared, dropped


def prepare_cicids2017_external_data(
    data_dir: str | Path,
    profile_path: str | Path,
    output_path: str | Path,
    chunksize: int = 100_000,
) -> dict[str, object]:
    """Create a deterministic, label-preserving CIC-IDS2017 external dataset.

    This command performs only header aliasing, numeric coercion, and removal of
    non-finite rows. It never invokes a model or computes evaluation metrics.
    """
    profile = load_release_profile(profile_path)
    fixed_features = profile["preprocessing"]["feature_selection"].get("fixed_features")
    if not fixed_features:
        raise ValueError("External preparation requires a profile with fixed_features.")
    directory = Path(data_dir)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing external dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    rows_dropped = 0
    label_counts = {"benign": 0, "attack": 0}
    wrote_header = False
    for source in files:
        for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            prepared, dropped = _standardize_cicids2017_chunk(chunk, fixed_features)
            prepared.to_csv(destination, index=False, mode="a", header=not wrote_header)
            wrote_header = True
            rows_written += len(prepared)
            rows_dropped += dropped
            label_counts["benign"] += int((prepared["label"] == 0).sum())
            label_counts["attack"] += int((prepared["label"] == 1).sum())

    if not rows_written:
        raise ValueError("External preparation produced no valid labeled rows.")
    manifest_path = destination.with_suffix(".manifest.json")
    manifest = {
        "workflow": "cicids2017_schema_standardization",
        "source_directory": str(directory),
        "source_files": [str(path) for path in files],
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "fixed_features": fixed_features,
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "rows_written": rows_written,
        "rows_dropped_nonfinite_or_unlabeled": rows_dropped,
        "label_counts": label_counts,
        "warning": (
            "This operation does not evaluate the model. Do not use CIC-IDS2017 labels "
            "for V14 model or threshold tuning before the configuration is frozen."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight external NIDS CSV schema compatibility")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Inspect external CSV headers without evaluation")
    preflight.add_argument("--data-dir", required=True, help="Directory containing external CSV files")
    preflight.add_argument("--model", required=True, help="Saved model artifact with feature schema")
    preflight.add_argument("--output", required=True, help="Directory for preflight reports")
    prepare = subparsers.add_parser("prepare-cicids2017", help="Standardize CIC-IDS2017 for a fixed feature schema")
    prepare.add_argument("--data-dir", required=True, help="Directory containing CIC-IDS2017 CSV files")
    prepare.add_argument("--config", required=True, help="V14 profile with fixed_features")
    prepare.add_argument("--output", required=True, help="New CSV path for standardized external data")
    prepare.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    if args.command == "preflight":
        report = preflight_external_directory(args.data_dir, args.model, args.output)
        compatible = int(report["schema_compatible"].sum())
        print(f"Schema-compatible files: {compatible}/{len(report)}")
    else:
        manifest = prepare_cicids2017_external_data(
            args.data_dir, args.config, args.output, args.chunksize
        )
        print(f"Prepared rows: {manifest['rows_written']}")


if __name__ == "__main__":
    main()
