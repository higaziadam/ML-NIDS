"""Tests for import-safe project configuration."""

import importlib
from pathlib import Path

import src.config as config


def test_config_import_does_not_create_directories(monkeypatch) -> None:
    """Read-only inference mounts must not fail merely by importing config."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Configuration import must not create directories")

    monkeypatch.setattr(Path, "mkdir", fail_if_called)
    importlib.reload(config)
