"""MRC and Fourier resampling checks for supported NumPy versions."""

import warnings

import mrcfile
import numpy as np
import pytest

from cryoflake.io import mrc
from cryoflake.io.mrc import (
    get_spectral_indices,
    load_mrc,
    make_cubic,
    normalize_voxel_size,
    normalized_grid_size,
    rescale_fourier,
    rescale_real,
    save_mrc,
    spectral_amplitude,
)
from cryoflake.preprocess.memory_guard import estimate_preprocess_memory


@pytest.mark.parametrize("shape", [(8, 10, 12), (5, 7, 9), (5, 8, 10)])
def test_spectral_shells_locate_dc_and_negative_frequencies(shape):
    spectrum = np.fft.rfftn(np.ones(shape, np.float32))
    radii = get_spectral_indices(spectrum)
    assert radii[0, 0, 0] == 0
    assert np.count_nonzero(radii == 0) == 1
    assert radii[-1, 0, 0] == 1
    assert radii[0, -1, 0] == 1
    assert spectral_amplitude(spectrum)[0] == np.prod(shape)


@pytest.mark.parametrize("shape", [(8, 10, 7), (5, 7, 4), (9, 6, 1), (3, 4, 0)])
def test_spectral_shell_broadcasting_preserves_exact_dense_shells(shape):
    z_grid, y_grid, x_grid = np.meshgrid(
        np.fft.fftfreq(shape[0]) * shape[0],
        np.fft.fftfreq(shape[1]) * shape[1],
        np.arange(shape[2]),
        indexing="ij",
    )
    expected = np.round(np.sqrt(x_grid**2 + y_grid**2 + z_grid**2))
    actual = get_spectral_indices(np.empty(shape, dtype=np.complex64))
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_fourier_resampling_round_trip_has_no_numpy_deprecations():
    z, y, x = np.indices((16, 16, 16))
    density = (
        3
        + np.sin(2 * np.pi * x / 16)
        + np.cos(2 * np.pi * y / 16)
        + np.sin(2 * np.pi * z / 16)
    ).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        downsampled = rescale_real(density, 8)
        restored = rescale_real(downsampled, 16)
    assert downsampled.shape == (8, 8, 8)
    assert restored.shape == density.shape
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, density, atol=2e-6)


@pytest.mark.parametrize("output_size", [4, 8])
def test_fourier_cropping_does_not_allocate_unused_zero_volume(
    monkeypatch, output_size
):
    spectrum = (
        np.arange(8 * 8 * 5, dtype=np.float32).reshape(8, 8, 5).astype(np.complex64)
    )
    shifted = np.fft.ifftshift(spectrum, axes=(0, 1))
    center = output_size // 2
    expected = np.fft.ifftshift(
        shifted[4 - center : 4 + center, 4 - center : 4 + center, : center + 1],
        axes=(0, 1),
    )

    def unexpected_zero_volume(*args, **kwargs):
        pytest.fail("Cropping or retaining a Fourier volume needs no zero padding")

    monkeypatch.setattr(mrc.np, "zeros", unexpected_zero_volume)
    actual = rescale_fourier(spectrum, output_size)
    assert actual.dtype == spectrum.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target", [8.0, 16.0, 1e5])
def test_coarse_positive_voxel_targets_retain_two_samples_per_axis(target):
    output, voxel = normalize_voxel_size(np.ones((8, 8, 8), np.float32), 1, target)
    assert output.shape == (2, 2, 2)
    assert voxel == 4.0
    assert np.isfinite(output).all()


@pytest.mark.parametrize(
    "argument,value",
    [
        (argument, value)
        for argument in ("input_voxel_size", "target_voxel_size")
        for value in (0, -1, np.nan, np.inf, -np.inf)
    ],
)
def test_voxel_normalization_rejects_invalid_sampling(argument, value):
    arguments = {"input_voxel_size": 1, "target_voxel_size": 1, argument: value}
    with pytest.raises(
        ValueError, match=f"{argument} must be a positive finite number"
    ):
        normalize_voxel_size(np.zeros((8, 8, 8), np.float32), **arguments)


@pytest.mark.parametrize(
    "shape", [(8, 8), (8, 8, 8, 8), (0, 0, 0), (8, 8, 6), (7, 7, 7)]
)
def test_normalization_requires_a_nonempty_even_cubic_3d_volume(shape):
    density = np.zeros(shape, np.float32)
    with pytest.raises(ValueError, match="nonempty even cubic 3D"):
        normalize_voxel_size(density, 1)
    with pytest.raises(ValueError, match="nonempty even cubic 3D"):
        rescale_real(density, 8)


@pytest.mark.parametrize("output_size", [0, -2, 1, 3, 2.5, True])
def test_fourier_resampling_rejects_invalid_output_sizes(output_size):
    with pytest.raises(ValueError, match="positive even integer"):
        rescale_real(np.zeros((8, 8, 8), np.float32), output_size)
    with pytest.raises(ValueError, match="positive even integer"):
        rescale_fourier(np.zeros((8, 8, 5), np.complex64), output_size)


@pytest.mark.parametrize("target,expected", [(1.0, 8), (1.6, 6), (2.5, 4), (3.0, 2)])
def test_normalized_grid_size_chooses_closest_even_sampling(target, expected):
    assert normalized_grid_size(8, 1, target) == expected


def test_memory_admission_estimate_matches_coarse_resampling(tmp_path):
    path = tmp_path / "density.mrc"
    save_mrc(np.ones((8, 8, 8), np.float32), 1, np.zeros(3), path)
    estimate = estimate_preprocess_memory(path, target_voxel_size=1e5)
    cubic, _, _ = make_cubic(np.ones((8, 8, 8), np.float32))
    output, _ = normalize_voxel_size(cubic, 1, 1e5)
    assert estimate.output_size == output.shape[0] == 2
    assert estimate.estimated_peak_bytes > 0


def test_mrc_round_trip_preserves_grid_and_optional_origin_units(tmp_path):
    path = tmp_path / "density.mrc"
    grid = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    origin = np.asarray([2, 4, 6], np.float32)
    save_mrc(grid, 2, origin, path)
    loaded = load_mrc(path, multiply_global_origin=False)
    np.testing.assert_array_equal(loaded.grid, grid)
    np.testing.assert_array_equal(loaded.global_origin, origin)
    assert loaded.voxel_size == 2
    assert loaded.grid.dtype == np.float32
    np.testing.assert_array_equal(load_mrc(path).global_origin, origin * 2)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.float32])
@pytest.mark.parametrize("axes", [(1, 2, 3), (3, 2, 1), (2, 1, 3)])
def test_mrc_conversion_owns_one_contiguous_float32_grid(tmp_path, dtype, axes):
    path = tmp_path / "density.mrc"
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    with mrcfile.new(path) as volume:
        volume.set_data(source)
        volume.voxel_size = 1
        volume.header["mapc"], volume.header["mapr"], volume.header["maps"] = axes
    if axes == (3, 2, 1):
        expected = np.moveaxis(source, [0, 1, 2], [2, 1, 0])
    elif axes == (2, 1, 3):
        expected = np.moveaxis(source, [1, 2, 0], [2, 1, 0])
    else:
        expected = source
    loaded = load_mrc(path)
    assert loaded.grid.dtype == np.float32
    assert loaded.grid.flags.c_contiguous
    assert loaded.grid.flags.owndata
    np.testing.assert_array_equal(loaded.grid, expected)


def test_mrc_reader_rejects_nonfinite_header_sampling(tmp_path):
    path = tmp_path / "invalid.mrc"
    with mrcfile.new(path) as volume:
        volume.set_data(np.ones((2, 2, 2), np.float32))
        volume.voxel_size = np.nan
    with pytest.raises(RuntimeError, match="does not have a header"):
        load_mrc(path)
