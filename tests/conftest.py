"""Shared fixtures built from the offline EMD-11736 / 7AE1 standard target."""

from __future__ import annotations

import pytest
from standard_target import load_standard_artifact_mapping, write_standard_structure


@pytest.fixture
def standard_artifact_mapping():
    return load_standard_artifact_mapping()


@pytest.fixture
def standard_structure_path(tmp_path):
    return write_standard_structure(tmp_path / "7ae1.cif")
