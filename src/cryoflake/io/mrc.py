"""CPU-only cryo-EM map loading and preprocessing primitives."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import mrcfile
import numpy as np

MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])


def load_mrc(
    mrc_path: str | Path,
    *,
    multiply_global_origin: bool = True,
) -> MRCObject:
    """Load a supported MRC axis arrangement without importing model code."""

    with mrcfile.open(mrc_path, "r") as mrc_file:
        voxel_size = float(mrc_file.voxel_size.x)
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise RuntimeError(
                f"Seems like the MRC file {mrc_path!s} does not have a header."
            )

        column_axis = int(mrc_file.header["mapc"])
        row_axis = int(mrc_file.header["mapr"])
        section_axis = int(mrc_file.header["maps"])
        origin = mrc_file.header["origin"]
        global_origin = np.array(
            [origin.x, origin.y, origin.z],
            dtype=np.float32,
        )
        global_origin += np.array(
            [
                mrc_file.header["nxstart"],
                mrc_file.header["nystart"],
                mrc_file.header["nzstart"],
            ],
            dtype=np.float32,
        )
        if multiply_global_origin:
            global_origin *= voxel_size

        source = np.asarray(mrc_file.data)
        axes = (column_axis, row_axis, section_axis)
        if axes == (1, 2, 3):
            grid = source
        elif axes == (3, 2, 1):
            grid = np.moveaxis(source, [0, 1, 2], [2, 1, 0])
        elif axes == (2, 1, 3):
            grid = np.moveaxis(source, [1, 2, 0], [2, 1, 0])
        else:
            raise RuntimeError(f"MRC axis arrangement {axes!r} is not supported")
        grid = np.array(grid, dtype=np.float32, order="C", copy=True)

    return MRCObject(grid, voxel_size, global_origin)


def save_mrc(
    grid: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    path: str | Path,
) -> None:
    grid = np.asarray(grid, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as output:
        output.set_data(grid)
        output.voxel_size = float(voxel_size)
        output.header["origin"].x = float(origin[0])
        output.header["origin"].y = float(origin[1])
        output.header["origin"].z = float(origin[2])
        output.update_header_stats()


def get_spectral_indices(fourier_grid: np.ndarray) -> np.ndarray:
    """Return rounded frequency-shell radii for an unshifted real FFT grid."""
    z_size, y_size, x_size = fourier_grid.shape
    z_squared = (np.fft.fftfreq(z_size) * z_size) ** 2
    y_squared = (np.fft.fftfreq(y_size) * y_size) ** 2
    x_squared = np.arange(x_size) ** 2
    # Keep the original x/y/z addition order without materializing three
    # coordinate volumes or separate square-root/rounding outputs.
    plane = x_squared[None, :] + y_squared[:, None]
    radii = plane[None, :, :] + z_squared[:, None, None]
    np.sqrt(radii, out=radii)
    np.round(radii, out=radii)
    return radii


def fourier_shell_average(
    values: np.ndarray,
    spectral_indices: np.ndarray | None = None,
) -> np.ndarray:
    if spectral_indices is None:
        spectral_indices = get_spectral_indices(values)
    spectral_indices = np.round(spectral_indices).astype(int)
    result = np.zeros(values.shape[-1], dtype=np.float64)
    for radius in range(values.shape[-1]):
        selected = spectral_indices == radius
        if np.any(selected):
            result[radius] = np.sum(values[selected]) / np.sum(selected)
    return result


def spectral_amplitude(
    fourier_grid: np.ndarray,
    spectral_indices: np.ndarray | None = None,
) -> np.ndarray:
    return fourier_shell_average(
        np.abs(fourier_grid),
        spectral_indices,
    )


def make_cubic(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(grid)
    if grid.ndim != 3 or min(grid.shape) <= 0:
        raise ValueError("Density must be a nonempty 3D volume")
    original_shape = np.asarray(grid.shape)
    size = max(max(grid.shape), 128)
    size += size % 2
    if np.all(original_shape == size):
        return grid, np.zeros(3, dtype=np.int64), original_shape
    output = np.zeros((size, size, size), dtype=grid.dtype)
    shift = np.asarray(output.shape) // 2 - original_shape // 2
    output[
        shift[0] : shift[0] + original_shape[0],
        shift[1] : shift[1] + original_shape[1],
        shift[2] : shift[2] + original_shape[2],
    ] = grid
    return output, shift, shift + original_shape


def rescale_fourier(fourier_grid: np.ndarray, output_size: int) -> np.ndarray:
    output_size = _even_output_size(output_size)
    fourier_grid = np.asarray(fourier_grid)
    if (
        fourier_grid.ndim != 3
        or min(fourier_grid.shape) <= 0
        or fourier_grid.shape[0] != fourier_grid.shape[1]
        or fourier_grid.shape[1] != (fourier_grid.shape[2] - 1) * 2
    ):
        raise ValueError("Input Fourier grid must represent a cubic volume")

    shifted = np.fft.ifftshift(fourier_grid, axes=(0, 1))
    input_center = shifted.shape[0] // 2
    output_center = output_size // 2
    if output_center < input_center:
        output = shifted[
            input_center - output_center : input_center + output_center,
            input_center - output_center : input_center + output_center,
            : output_center + 1,
        ]
    elif output_center > input_center:
        output = np.zeros(
            (output_size, output_size, output_center + 1),
            dtype=fourier_grid.dtype,
        )
        output[
            output_center - input_center : output_center + input_center,
            output_center - input_center : output_center + input_center,
            : shifted.shape[2],
        ] = shifted
    else:
        output = shifted
    return np.fft.ifftshift(output, axes=(0, 1))


def rescale_real(grid: np.ndarray, output_size: int) -> np.ndarray:
    grid = _even_cubic_density(grid)
    output_size = _even_output_size(output_size)
    if output_size == grid.shape[0]:
        return grid
    fourier_grid = np.fft.rfftn(grid, axes=(0, 1, 2))
    return np.fft.irfftn(
        rescale_fourier(fourier_grid, output_size),
        s=(output_size, output_size, output_size),
        axes=(0, 1, 2),
    )


def _positive_voxel_size(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _even_output_size(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Output size must be a positive even integer")
    if value < 2 or value % 2:
        raise ValueError("Output size must be a positive even integer")
    return int(value)


def _even_cubic_density(density: np.ndarray) -> np.ndarray:
    density = np.asarray(density)
    if (
        density.ndim != 3
        or len(set(density.shape)) != 1
        or min(density.shape) < 2
        or density.shape[0] % 2
    ):
        raise ValueError("Density must be a nonempty even cubic 3D volume")
    return density


def normalized_grid_size(
    input_size: int,
    input_voxel_size: float,
    target_voxel_size: float,
) -> int:
    """Choose an even resampled size, retaining at least two samples per axis.

    The actual voxel size is the closest supported value to the requested size;
    the memory admission estimate uses this same calculation.
    """
    input_size = _even_output_size(input_size)
    input_voxel_size = _positive_voxel_size(input_voxel_size, "input_voxel_size")
    target_voxel_size = _positive_voxel_size(target_voxel_size, "target_voxel_size")
    requested_size = input_size * input_voxel_size / target_voxel_size
    if not np.isfinite(requested_size):
        raise ValueError("Voxel size ratio is too large to resample")
    output_size = max(2, int(round(requested_size)))
    if output_size % 2:
        upper_size = output_size + 1
        lower_size = output_size - 1
        upper_voxel = input_voxel_size * input_size / upper_size
        lower_voxel = input_voxel_size * input_size / lower_size
        output_size = (
            upper_size
            if abs(upper_voxel - target_voxel_size)
            < abs(lower_voxel - target_voxel_size)
            else lower_size
        )
    return output_size


def normalize_voxel_size(
    density: np.ndarray,
    input_voxel_size: float,
    target_voxel_size: float = 1.0,
) -> tuple[np.ndarray, float]:
    density = _even_cubic_density(density)
    input_voxel_size = _positive_voxel_size(input_voxel_size, "input_voxel_size")
    output_size = normalized_grid_size(
        density.shape[0], input_voxel_size, target_voxel_size
    )
    output_voxel_size = input_voxel_size * density.shape[0] / output_size
    return rescale_real(density, output_size), output_voxel_size


def prepare_grid(
    grid: np.ndarray,
    voxel_size: float,
    global_origin: np.ndarray,
    target_voxel_size: float = 1.5,
) -> MRCObject:
    voxel_size = _positive_voxel_size(voxel_size, "voxel_size")
    target_voxel_size = _positive_voxel_size(target_voxel_size, "target_voxel_size")
    cubic_grid, shift, _ = make_cubic(grid)
    adjusted_origin = np.asarray(global_origin, dtype=np.float32).copy()
    adjusted_origin -= shift * float(voxel_size)
    normalized_grid, normalized_voxel_size = normalize_voxel_size(
        cubic_grid,
        voxel_size,
        target_voxel_size=target_voxel_size,
    )
    return MRCObject(
        np.asarray(normalized_grid, dtype=np.float32),
        normalized_voxel_size,
        adjusted_origin,
    )
