import math
import os
from pathlib import Path
from typing import BinaryIO, Callable, Mapping

import numpy as np

try:
    from numpy.lib._format_impl import _read_array_header
except ImportError:
    # NumPy 1.x keeps this parser in format. Public header readers do not
    # support NPY 3's UTF-8 structured field names.
    from numpy.lib.format import _read_array_header

from cryoflake.sequence.validation import (
    validate_sequence_embedding_payload,
)
from cryoflake.storage.store import (
    replace_preprocessed_artifact_sequence,
)


def _validate_ignored_npz_member(member: BinaryIO, max_header_size: int) -> None:
    expected_payload_size = None
    if member.read(len(np.lib.format.MAGIC_PREFIX)) == np.lib.format.MAGIC_PREFIX:
        member.seek(0)
        version = np.lib.format.read_magic(member)
        shape, _, dtype = _read_array_header(
            member, version, max_header_size=max_header_size
        )
        if dtype.hasobject:
            raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
        # Validate dimensions without allocating the ignored array's payload.
        np.broadcast_to(np.uint8(0), shape)
        expected_payload_size = math.prod(shape) * dtype.itemsize

    # Ignored members still undergo decompression and CRC validation. Consume
    # them in bounded chunks, preserving raw non-NPY members and trailing data.
    payload_size = 0
    while chunk := member.read(256 * 1024):
        payload_size += len(chunk)
    if expected_payload_size is not None and payload_size < expected_payload_size:
        raise ValueError("NPY array data is truncated")


def _read_npz_payload(sequence_embedding_path: Path) -> dict[str, object]:
    supported_keys = {"sequence", "sequence_mask", "unified_seq", "unified_seq_len"}
    payload = {}
    with np.load(sequence_embedding_path, allow_pickle=False) as npz_file:
        members = npz_file.zip.infolist()
        members_by_name = {member.filename: member for member in members}
        supported_members = {}
        for key in supported_keys:
            # Preserve NumPy's exact-name preference and ZIP's last-duplicate
            # selection, while identifying the actual member that is loaded.
            member_name = key if key in members_by_name else f"{key}.npy"
            if member_name in members_by_name:
                supported_members[members_by_name[member_name]] = key
        for member_info in members:
            key = supported_members.get(member_info)
            if key is not None:
                payload[key] = npz_file[key]
            else:
                # Open each ignored physical entry, including members hidden
                # by aliases or duplicate filenames, rather than resolving its
                # name again and accidentally rereading the selected member.
                with npz_file.zip.open(member_info) as member:
                    _validate_ignored_npz_member(member, npz_file.max_header_size)
    return payload


def load_sequence_embedding_file(
    sequence_embedding_path: os.PathLike,
) -> dict[str, np.ndarray]:
    """Load a numeric NPY array or NPZ input with the ``sequence`` field.

    NPZ inputs may also contain ``sequence_mask`` and unified sequence metadata.
    Object arrays are rejected without deserialization.
    """
    sequence_embedding_path = Path(sequence_embedding_path)
    suffix = sequence_embedding_path.suffix.lower()

    if suffix == ".npy":
        payload = {"sequence": np.load(sequence_embedding_path, allow_pickle=False)}
    elif suffix == ".npz":
        payload = _read_npz_payload(sequence_embedding_path)
    else:
        raise ValueError(
            "Unsupported sequence embedding file type: "
            f"{sequence_embedding_path.suffix}. Expected .npy or .npz."
        )

    return validate_sequence_embedding_payload(payload)


def attach_sequence_embeddings_to_preprocessed_artifact(
    artifact_reference: os.PathLike | str,
    sequence_embedding_path: os.PathLike,
    sequence_embedding_loader: Callable[[os.PathLike], Mapping[str, np.ndarray]] = (
        load_sequence_embedding_file
    ),
    s3_endpoint_url: str | None = None,
) -> None:
    """Load an embedding sidecar and replace sequence fields in an artifact."""
    sequence_payload = dict(sequence_embedding_loader(sequence_embedding_path))
    replace_preprocessed_artifact_sequence(
        artifact_reference,
        sequence_payload,
        s3_endpoint_url=s3_endpoint_url,
    )
