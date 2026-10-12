"""Data-directory resolution.

The default derives the data dir from this source file's location, which only
holds for a repo checkout. Installed into site-packages (as in the Docker image)
that path points nowhere, so BIORAG_DATA_DIR must be able to override it.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import biorag.config as config


@pytest.fixture
def reload_config(monkeypatch):
    def _reload(value: str | None):
        if value is None:
            monkeypatch.delenv("BIORAG_DATA_DIR", raising=False)
        else:
            monkeypatch.setenv("BIORAG_DATA_DIR", value)
        return importlib.reload(config)

    yield _reload
    # Paths are module globals: restore them so later tests see the defaults.
    monkeypatch.delenv("BIORAG_DATA_DIR", raising=False)
    importlib.reload(config)


def test_default_data_dir_is_repo_relative(reload_config):
    cfg = reload_config(None)
    assert cfg.DATA_DIR == cfg.PROJECT_ROOT / "data"


def test_env_var_overrides_data_dir_and_children(reload_config, tmp_path):
    cfg = reload_config(str(tmp_path))
    assert cfg.DATA_DIR == tmp_path
    assert cfg.INDEX_DIR == tmp_path / "index"
    assert cfg.RAW_DIR == tmp_path / "raw"
    assert cfg.CLEAN_DIR == tmp_path / "clean"


def test_empty_env_var_falls_back_to_default(reload_config):
    cfg = reload_config("")
    assert cfg.DATA_DIR == Path(cfg.PROJECT_ROOT) / "data"
