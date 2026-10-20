"""Metadata validators enforce the values stored in float32 artifacts."""

import re
from types import MappingProxyType

import numpy as np
import pytest

import cryoflake
from cryoflake import metadata
from cryoflake.metadata import (
    get_preprocessed_artifact_global_origin,
    get_preprocessed_artifact_grid,
    get_preprocessed_artifact_voxel_size,
    validate_coordinates,
    validate_density_grid,
    validate_global_origin,
    validate_preprocessed_artifact_mapping,
)


def test_artifact_mapping_validator_has_one_canonical_public_export():
    assert cryoflake.validate_preprocessed_artifact_mapping is (
        metadata.validate_preprocessed_artifact_mapping
    )
    assert "validate_preprocessed_artifact_mapping" in cryoflake.__all__
    assert not hasattr(cryoflake, "validate_preprocessed_example_mapping")
    assert not hasattr(metadata, "validate_preprocessed_example_mapping")


def test_artifact_mapping_validator_preserves_mapping_identity_and_extra_fields():
    artifact_mapping = MappingProxyType({"application_field": object()})

    actual = validate_preprocessed_artifact_mapping(artifact_mapping=artifact_mapping)

    assert actual is artifact_mapping


@pytest.mark.parametrize(
    "validator",
    [
        validate_preprocessed_artifact_mapping,
        get_preprocessed_artifact_grid,
        get_preprocessed_artifact_global_origin,
        get_preprocessed_artifact_voxel_size,
    ],
)
@pytest.mark.parametrize("source", [None, "artifacts/example/metadata.json"])
def test_artifact_mapping_rejection_reports_artifact_and_source(validator, source):
    label = "preprocessed artifact"
    if source is not None:
        label += f" in {source}"
    with pytest.raises(ValueError, match=re.escape(label + " must be a mapping.")):
        validator(artifact_mapping=[], source=source)


@pytest.mark.parametrize(
    "getter, artifact_mapping, expected",
    [
        (
            get_preprocessed_artifact_grid,
            {"density_grid": np.arange(8).reshape(2, 2, 2)},
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        ),
        (
            get_preprocessed_artifact_global_origin,
            {"global_origin": [1.5, 2.5, 3.5]},
            np.asarray([1.5, 2.5, 3.5], dtype=np.float32),
        ),
        (get_preprocessed_artifact_voxel_size, {"voxel_size": 1.5}, 1.5),
    ],
)
def test_geometry_getters_accept_artifact_mapping_keyword(
    getter, artifact_mapping, expected
):
    actual = getter(artifact_mapping=artifact_mapping, source="artifact fixture")

    np.testing.assert_array_equal(actual, expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    "validator, values",
    [
        (validate_density_grid, np.asarray([[[1e100]]])),
        (validate_coordinates, np.asarray([[1e100, 0, 0]])),
        (validate_global_origin, np.asarray([1e100, 0, 0])),
    ],
)
def test_metadata_rejects_values_that_overflow_float32(validator, values):
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="finite"):
        validator(values)


def test_coordinate_nan_support_does_not_allow_float32_overflow():
    coordinates = np.asarray([[np.nan, np.nan, np.nan], [1e100, 0, 0]])
    with pytest.raises(ValueError, match="infinite coordinates"):
        validate_coordinates(coordinates, allow_nan=True)


@pytest.mark.parametrize(
    "validator, values",
    [
        (validate_density_grid, np.arange(8, dtype=np.float64).reshape(2, 2, 2)),
        (validate_coordinates, np.asarray([[1.5, 2.5, 3.5]])),
        (validate_global_origin, np.asarray([1.5, 2.5, 3.5])),
    ],
)
def test_metadata_converts_representable_values_without_mutating_inputs(
    validator, values
):
    original = values.copy()
    converted = validator(values)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, values)
    np.testing.assert_array_equal(values, original)


def test_coordinate_nan_support_preserves_missing_atoms():
    coordinates = np.asarray([[np.nan, np.nan, np.nan], [1, 2, 3]])
    converted = validate_coordinates(coordinates, allow_nan=True)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, coordinates)


@pytest.mark.parametrize(
    "coordinates",
    [
        np.empty((0, 4)),
        np.empty((2, 0)),
        np.empty((0, 3), dtype=str),
        np.empty((0, 3), dtype=object),
    ],
)
def test_empty_coordinates_still_require_numeric_coordinate_shape(coordinates):
    with pytest.raises(ValueError, match="coordinate|shape"):
        validate_coordinates(coordinates)


@pytest.mark.parametrize("coordinates", [[], np.empty((0, 3))])
def test_empty_coordinate_lists_and_arrays_remain_valid(coordinates):
    actual = validate_coordinates(coordinates)
    assert actual.shape == (0, 3)
    assert actual.dtype == np.float32
