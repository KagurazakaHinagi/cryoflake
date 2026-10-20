import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional
from urllib.request import urlopen

import numpy as np

from cryoflake.storage.transports import (
    is_remote_uri,
    resolve_artifact_list_entry,
)

WWPDB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]{4}$")
WWPDB_ID_SEARCH_PATTERN = re.compile(r"\b[a-zA-Z0-9]{4}\b")
SOURCE_CSV_EMDB_COLUMN = "EMD-"
SOURCE_CSV_ASSOCIATED_PDB_COLUMN = "Associated PDB"
SOURCE_CSV_RESOLUTION_COLUMN_PREFIX = "resolution"
WWPDB_PLACEHOLDER_VALUES = {"none", "null", "n/a", "na"}


@dataclass(frozen=True)
class ManifestEntry:
    emdb_id: str
    wwpdb_id: Optional[str]
    source_resolution: str = ""

    @property
    def emdb_accession(self) -> str:
        return f"EMD-{self.emdb_id}"

    @property
    def map_url(self) -> str:
        return get_map_url(self.emdb_id)

    @property
    def structure_url(self) -> str:
        if self.wwpdb_id is None:
            return ""
        return get_structure_url(self.wwpdb_id)

    @property
    def map_file_name(self) -> str:
        return f"emd_{self.emdb_id}.map.gz"

    @property
    def structure_file_name(self) -> str:
        if self.wwpdb_id is None:
            return ""
        return f"{self.wwpdb_id}.cif.gz"


def normalize_emdb_id(emdb_id: str) -> str:
    normalized_id = emdb_id.strip().upper()
    if normalized_id.startswith("EMD-"):
        normalized_id = normalized_id[4:]
    if not re.match(r"^\d{4,5}$", normalized_id):
        raise ValueError(f"Invalid EMDB ID: {emdb_id}")
    return normalized_id


def normalize_wwpdb_id(wwpdb_id: str) -> str:
    normalized_id = wwpdb_id.strip().lower()
    if not WWPDB_ID_PATTERN.match(normalized_id):
        raise ValueError(f"Invalid wwPDB ID: {wwpdb_id}")
    return normalized_id


def get_map_url(emdb_id: str) -> str:
    normalized_id = normalize_emdb_id(emdb_id)
    return (
        "https://files.wwpdb.org/pub/emdb/structures/"
        f"EMD-{normalized_id}/map/emd_{normalized_id}.map.gz"
    )


def get_structure_url(wwpdb_id: str) -> str:
    normalized_id = normalize_wwpdb_id(wwpdb_id)
    return (
        "https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/"
        f"{normalized_id[1:3]}/{normalized_id}.cif.gz"
    )


def get_first_associated_wwpdb_id(associated_pdb: str) -> Optional[str]:
    value = associated_pdb.strip()
    if not value or value.lower() in WWPDB_PLACEHOLDER_VALUES:
        return None

    for match in WWPDB_ID_SEARCH_PATTERN.finditer(value):
        candidate = match.group(0)
        if candidate.lower() in WWPDB_PLACEHOLDER_VALUES:
            continue
        return normalize_wwpdb_id(candidate)

    return None


def _normalized_csv_row(row: Mapping[Optional[str], Optional[str]]) -> dict[str, str]:
    return {
        fieldname.strip(): (value or "").strip()
        for fieldname, value in row.items()
        if fieldname is not None
    }


def read_source_associations(source_csv: os.PathLike) -> list[ManifestEntry]:
    required_fieldnames = {
        SOURCE_CSV_EMDB_COLUMN,
        SOURCE_CSV_ASSOCIATED_PDB_COLUMN,
    }
    entries = []
    seen_emdb_ids = set()

    with Path(source_csv).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = [
            fieldname.strip()
            for fieldname in (reader.fieldnames or [])
            if fieldname is not None
        ]
        missing_fieldnames = required_fieldnames - set(fieldnames)
        if missing_fieldnames:
            missing = ", ".join(sorted(missing_fieldnames))
            raise ValueError(f"Source CSV is missing required columns: {missing}")
        resolution_fieldname = next(
            (
                fieldname
                for fieldname in fieldnames
                if fieldname.lower().startswith(SOURCE_CSV_RESOLUTION_COLUMN_PREFIX)
            ),
            None,
        )

        for row in reader:
            normalized_row = _normalized_csv_row(row)
            if not any(normalized_row.values()):
                continue
            emdb_id = normalize_emdb_id(normalized_row[SOURCE_CSV_EMDB_COLUMN])
            if emdb_id in seen_emdb_ids:
                continue
            seen_emdb_ids.add(emdb_id)
            wwpdb_id = get_first_associated_wwpdb_id(
                normalized_row[SOURCE_CSV_ASSOCIATED_PDB_COLUMN]
            )
            source_resolution = ""
            if resolution_fieldname is not None:
                source_resolution = normalized_row.get(resolution_fieldname, "")
            entries.append(
                ManifestEntry(
                    emdb_id=emdb_id,
                    wwpdb_id=wwpdb_id,
                    source_resolution=source_resolution,
                )
            )

    return entries


def fetch_json(url: str, timeout: float = 30.0) -> Mapping:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def get_fitted_structure_id_from_emdb_entry(
    emdb_id: str,
    json_fetcher: Callable[[str], Mapping] = fetch_json,
) -> Optional[str]:
    normalized_id = normalize_emdb_id(emdb_id)
    request_url = f"https://www.ebi.ac.uk/emdb/api/entry/fitted/{normalized_id}"
    payload = json_fetcher(request_url)

    crossreferences = payload.get("crossreferences") or {}
    pdb_list = crossreferences.get("pdb_list") or {}
    pdb_references = pdb_list.get("pdb_reference") or []
    if isinstance(pdb_references, Mapping):
        pdb_references = [pdb_references]

    for pdb_reference in pdb_references:
        pdb_id = pdb_reference.get("pdb_id")
        if pdb_id:
            return normalize_wwpdb_id(pdb_id)

    return None


def build_manifest(
    source_csv: os.PathLike,
    include_without_structure: bool = False,
) -> list[ManifestEntry]:
    entries = []
    for entry in read_source_associations(source_csv):
        if entry.wwpdb_id is None and not include_without_structure:
            continue
        entries.append(entry)
    return entries


def _resolve_map_path(entry: ManifestEntry, maps_dir: os.PathLike) -> Path:
    maps_dir = Path(maps_dir)
    local_candidates = [
        maps_dir / f"emd_{entry.emdb_id}.map",
        maps_dir / f"emd_{entry.emdb_id}.mrc",
    ]
    existing_candidates = [path for path in local_candidates if path.is_file()]
    if len(existing_candidates) > 1:
        candidates = ", ".join(str(path) for path in existing_candidates)
        raise ValueError(
            f"Ambiguous local density map for {entry.emdb_accession}: "
            f"found {candidates}. Keep only one .map or .mrc file."
        )
    if existing_candidates:
        return existing_candidates[0]
    return maps_dir / entry.map_file_name


def _resolve_structure_path(entry: ManifestEntry, structures_dir: os.PathLike) -> Path:
    structures_dir = Path(structures_dir)
    local_cif_path = structures_dir / f"{entry.wwpdb_id}.cif"
    if local_cif_path.is_file():
        return local_cif_path
    return structures_dir / entry.structure_file_name


def manifest_rows(
    entries: Iterable[ManifestEntry],
    maps_dir: Optional[os.PathLike] = None,
    structures_dir: Optional[os.PathLike] = None,
    sequence_embeddings_dir: Optional[os.PathLike] = None,
    sequence_embedding_suffix: str = ".npz",
    sequence_embedding_name_template: str = "{wwpdb_id}",
) -> list[dict[str, str]]:
    if sequence_embedding_suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("sequence_embedding_suffix must be .npy or .npz")
    rows = []
    for entry in entries:
        map_path = entry.map_file_name
        structure_path = entry.structure_file_name
        sequence_embedding_path = ""
        if maps_dir is not None:
            map_path = str(_resolve_map_path(entry, maps_dir))
        if structures_dir is not None and entry.structure_file_name:
            structure_path = str(_resolve_structure_path(entry, structures_dir))
        if sequence_embeddings_dir is not None:
            artifact_name = Path(
                get_preprocessed_artifact_name(
                    {"emdb_id": entry.emdb_id, "wwpdb_id": entry.wwpdb_id or ""}
                )
            ).stem
            sequence_embedding_file_name = sequence_embedding_name_template.format(
                emdb_id=entry.emdb_id,
                emdb_accession=entry.emdb_accession,
                artifact_name=artifact_name,
                wwpdb_id=entry.wwpdb_id or "",
            )
            if not sequence_embedding_file_name:
                raise ValueError(
                    "sequence_embedding_name_template produced an empty file name "
                    f"for {entry.emdb_accession}."
                )
            sequence_embedding_path = str(
                Path(sequence_embeddings_dir)
                / f"{sequence_embedding_file_name}{sequence_embedding_suffix}"
            )
        row = {
            "emdb_accession": entry.emdb_accession,
            "emdb_id": entry.emdb_id,
            "wwpdb_id": entry.wwpdb_id or "",
            "source_resolution": entry.source_resolution,
            "map_url": entry.map_url,
            "structure_url": entry.structure_url,
            "map_path": map_path,
            "structure_path": structure_path,
        }
        if sequence_embeddings_dir is not None:
            row["sequence_embedding_path"] = sequence_embedding_path
        rows.append(row)
    return rows


def write_manifest_csv(
    entries: Iterable[ManifestEntry],
    manifest_csv: os.PathLike,
    maps_dir: Optional[os.PathLike] = None,
    structures_dir: Optional[os.PathLike] = None,
    sequence_embeddings_dir: Optional[os.PathLike] = None,
    sequence_embedding_suffix: str = ".npz",
    sequence_embedding_name_template: str = "{wwpdb_id}",
) -> None:
    output_path = Path(manifest_csv)
    manifest_dir = output_path.parent.resolve()
    rows = manifest_rows(
        entries,
        maps_dir=maps_dir,
        structures_dir=structures_dir,
        sequence_embeddings_dir=sequence_embeddings_dir,
        sequence_embedding_suffix=sequence_embedding_suffix,
        sequence_embedding_name_template=sequence_embedding_name_template,
    )
    for row in rows:
        for fieldname in ("map_path", "structure_path", "sequence_embedding_path"):
            path_value = row.get(fieldname, "")
            if not path_value:
                continue
            path = Path(path_value)
            if path.is_absolute():
                continue
            row[fieldname] = str(Path(os.path.relpath(path.resolve(), manifest_dir)))

    fieldnames = [
        "emdb_accession",
        "emdb_id",
        "wwpdb_id",
        "source_resolution",
        "map_url",
        "structure_url",
        "map_path",
        "structure_path",
    ]
    if sequence_embeddings_dir is not None:
        fieldnames.append("sequence_embedding_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest_csv(manifest_csv: os.PathLike) -> list[dict[str, str]]:
    required_fieldnames = {
        "emdb_id",
        "wwpdb_id",
        "map_path",
        "structure_path",
    }
    with Path(manifest_csv).open(newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = set(reader.fieldnames or [])
        missing_fieldnames = required_fieldnames - fieldnames
        if missing_fieldnames:
            missing = ", ".join(sorted(missing_fieldnames))
            raise ValueError(f"Manifest is missing required columns: {missing}")
        return [dict(row) for row in reader]


def get_preprocessed_artifact_name(manifest_row: Mapping[str, str]) -> str:
    """Return the extensionless root name for one preprocessed artifact."""

    emdb_id = normalize_emdb_id(manifest_row["emdb_id"])
    wwpdb_id = manifest_row.get("wwpdb_id", "").strip()
    if wwpdb_id:
        return f"emd_{emdb_id}_{normalize_wwpdb_id(wwpdb_id)}"
    return f"emd_{emdb_id}"


def write_artifact_list(
    artifact_references: Iterable[os.PathLike],
    artifact_list_path: os.PathLike,
) -> None:
    artifact_list_path = Path(artifact_list_path)
    artifact_list_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_list_dir = artifact_list_path.parent
    artifact_list_dir = artifact_list_dir.resolve()

    path_lines = []
    for path in artifact_references:
        if is_remote_uri(path):
            path_lines.append(str(path))
            continue
        path = Path(path)
        if path.is_absolute():
            path_lines.append(str(path))
            continue
        path_lines.append(str(Path(os.path.relpath(path.resolve(), artifact_list_dir))))

    artifact_list_path.write_text("\n".join(path_lines) + ("\n" if path_lines else ""))


def read_artifact_list(
    artifact_list_path: os.PathLike,
) -> list[Path | str]:
    """Read artifact roots, resolving local relative paths from the list."""

    artifact_list_path = Path(artifact_list_path)
    base_dir = artifact_list_path.parent
    paths = [
        resolve_artifact_list_entry(line.strip(), base_dir)
        for line in artifact_list_path.read_text().splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"Artifact list is empty: {artifact_list_path}")
    return paths


def split_artifact_references(
    artifact_references: Iterable[os.PathLike],
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[Path | str], list[Path | str]]:
    """Split unique artifact references, retaining the first spelling of each.

    A singleton stays in training: validation never contains an artifact that
    is also present in training. Root and metadata.json references identify
    the same artifact, as do local relative, absolute, and symlink paths.
    """
    validation_fraction = float(validation_fraction)
    if (
        not np.isfinite(validation_fraction)
        or validation_fraction <= 0
        or validation_fraction >= 1
    ):
        raise ValueError(
            "validation_fraction must be finite, greater than 0, and less than 1"
        )

    from cryoflake.storage.store import preprocessed_artifact_root
    from cryoflake.storage.transports import join_remote_uri, split_s3_uri

    paths: list[Path | str] = []
    seen_references: set[tuple] = set()
    for path in artifact_references:
        value = os.fspath(path)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Artifact references must be nonempty paths or S3 URIs")
        root = preprocessed_artifact_root(value)
        if is_remote_uri(root):
            bucket, key = split_s3_uri(join_remote_uri(root, "metadata.json"))
            identity = ("s3", bucket, key)
            if identity in seen_references:
                continue
            seen_references.add(identity)
            paths.append(value)
        else:
            local_root = Path(root).resolve()
            identity = ("local", str(local_root))
            if identity in seen_references:
                continue
            seen_references.add(identity)
            try:
                stat = local_root.stat()
            except FileNotFoundError:
                pass
            else:
                inode_identity = ("inode", stat.st_dev, stat.st_ino)
                if inode_identity in seen_references:
                    continue
                seen_references.add(inode_identity)
            paths.append(Path(value))
    if len(paths) == 0:
        return [], []
    if len(paths) == 1:
        return paths, []

    valid_count = round(len(paths) * validation_fraction)
    valid_count = min(max(1, valid_count), len(paths) - 1)

    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)
    valid_indices = set(indices[:valid_count])

    train_artifact_references = [
        path for index, path in enumerate(paths) if index not in valid_indices
    ]
    validation_artifact_references = [
        path for index, path in enumerate(paths) if index in valid_indices
    ]
    return train_artifact_references, validation_artifact_references
