"""Tests for independent-external-data schema preflight checks."""

import pandas as pd

from src.external_validation import inspect_external_csv


def test_cicids2017_aliases_are_normalized_but_protocol_remains_required(tmp_path) -> None:
    source = tmp_path / "cicids2017.csv"
    pd.DataFrame(
        {
            " Total Length of Fwd Packets": [10],
            " min_seg_size_forward": [1],
            " Init_Win_bytes_forward": [2],
            " Init_Win_bytes_backward": [3],
            " Max Packet Length": [4],
            " Min Packet Length": [5],
            " Label": ["BENIGN"],
        }
    ).to_csv(source, index=False)

    report = inspect_external_csv(
        source,
        [
            "Fwd Packets Length Total",
            "Fwd Seg Size Min",
            "Init Fwd Win Bytes",
            "Init Bwd Win Bytes",
            "Packet Length Max",
            "Packet Length Min",
            "Protocol",
        ],
    )

    assert report["label_column_present"] is True
    assert report["missing_required_features"] == "Protocol"
    assert report["schema_compatible"] is False
