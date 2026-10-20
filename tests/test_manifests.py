from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import pytest

from cryoflake.common import path_key
from cryoflake.manifests import download as download_module
from cryoflake.manifests import manifest
from cryoflake.manifests.download import download_manifest_entries
from cryoflake.manifests.manifest import (
    ManifestEntry,
    get_preprocessed_artifact_name,
    manifest_rows,
    read_manifest_csv,
    split_artifact_references,
    write_manifest_csv,
)
from cryoflake.manifests.splits import (
    GroupedSplitRecord,
    read_holdout_group_ids,
    split_grouped_records,
    write_grouped_split_report,
)


def test_concurrent_downloads_to_same_destination_use_isolated_files(
    monkeypatch, tmp_path
):
    ready = threading.Barrier(2)

    class Response(io.BytesIO):
        def read(self, *args):
            if self.tell() == 0:
                ready.wait(timeout=2)
            return super().read(*args)

    monkeypatch.setattr(download_module, "urlopen", lambda _: Response(b"complete"))
    destination = tmp_path / "map.mrc"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                download_module.download_url,
                "https://example.test/map",
                destination,
                True,
            )
            for _ in range(2)
        ]
        assert [future.result(timeout=5) for future in futures] == [destination] * 2
    assert destination.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [destination]


def test_interrupted_download_removes_partial_file_and_closes_response(
    monkeypatch, tmp_path
):
    class Response(io.BytesIO):
        def read(self, *args):
            if self.tell():
                raise KeyboardInterrupt("download interrupted")
            return super().read(2)

    response = Response(b"complete")
    monkeypatch.setattr(download_module, "urlopen", lambda _: response)
    destination = tmp_path / "map.mrc"
    destination.write_bytes(b"previous")
    with pytest.raises(KeyboardInterrupt, match="download interrupted"):
        download_module.download_url(
            "https://example.test/map", destination, force_download=True
        )
    assert response.closed
    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_http_download_error_closes_error_response(monkeypatch, tmp_path):
    body = io.BytesIO(b"not found")
    error = HTTPError("https://example.test/map", 404, "missing", None, body)

    def missing(_):
        raise error

    monkeypatch.setattr(download_module, "urlopen", missing)
    with pytest.raises(HTTPError):
        download_module.download_url("https://example.test/map", tmp_path / "map.mrc")
    assert body.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["single", "manifest"])
def test_download_force_keyword_does_not_keep_legacy_alias(tmp_path, operation):
    with pytest.raises(TypeError, match="force"):
        if operation == "single":
            download_module.download_url(
                "https://example.test/map", tmp_path / "map.mrc", force=True
            )
        else:
            download_manifest_entries(
                [], tmp_path / "maps", tmp_path / "structures", force=True
            )
    assert not list(tmp_path.iterdir())


def test_download_force_policy_refreshes_only_when_enabled(monkeypatch, tmp_path):
    calls = []

    def response(url):
        calls.append(url)
        return io.BytesIO(b"new data")

    monkeypatch.setattr(download_module, "urlopen", response)
    destination = tmp_path / "map.mrc"
    destination.write_bytes(b"reused data")
    assert (
        download_module.download_url("https://example.test/map", destination)
        == destination
    )
    assert destination.read_bytes() == b"reused data"
    assert calls == []
    assert (
        download_module.download_url(
            "https://example.test/map", destination, force_download=True
        )
        == destination
    )
    assert destination.read_bytes() == b"new data"
    assert calls == ["https://example.test/map"]


def test_failed_parallel_download_waits_for_active_transfer_cleanup(
    monkeypatch, tmp_path
):
    started = threading.Event()
    release = threading.Event()
    failed = threading.Event()
    cleaned_up = threading.Event()
    returned = threading.Event()
    errors = []
    entries = [ManifestEntry("1234", None), ManifestEntry("1235", None)]

    def download(url, output_path, force_download=False):
        if url == entries[0].map_url:
            assert started.wait(timeout=2)
            failed.set()
            raise RuntimeError("transfer failed")
        started.set()
        try:
            assert release.wait(timeout=2)
        finally:
            cleaned_up.set()
        return output_path

    def run():
        try:
            download_manifest_entries(
                entries, tmp_path / "maps", tmp_path / "structures", num_workers=2
            )
        except RuntimeError as exc:
            errors.append(exc)
        finally:
            returned.set()

    monkeypatch.setattr(download_module, "download_url", download)
    runner = threading.Thread(target=run)
    runner.start()
    try:
        assert failed.wait(timeout=2)
        assert not returned.wait(timeout=0.1)
    finally:
        release.set()
        runner.join(timeout=3)
    assert not runner.is_alive()
    assert cleaned_up.is_set()
    assert len(errors) == 1
    assert "transfer failed" in str(errors[0])


def test_manifest_embedding_paths_default_to_npz(tmp_path):
    entries = [ManifestEntry("1234", "1abc")]
    embeddings_dir = tmp_path / "embeddings"
    rows = manifest_rows(entries, sequence_embeddings_dir=embeddings_dir)
    assert rows[0]["sequence_embedding_path"] == str(embeddings_dir / "1abc.npz")

    manifest = tmp_path / "manifest.csv"
    write_manifest_csv(entries, manifest, sequence_embeddings_dir=embeddings_dir)
    assert read_manifest_csv(manifest)[0]["sequence_embedding_path"] == str(
        embeddings_dir / "1abc.npz"
    )


@pytest.mark.parametrize(
    "entry, expected",
    [
        (ManifestEntry("1234", "1ABC"), "emd_1234_1abc"),
        (ManifestEntry("12345", None), "emd_12345"),
    ],
)
def test_artifact_name_matches_manifest_sidecar_template(tmp_path, entry, expected):
    rows = manifest_rows(
        [entry],
        sequence_embeddings_dir=tmp_path,
        sequence_embedding_name_template="{artifact_name}",
    )
    row = rows[0]
    assert get_preprocessed_artifact_name(row) == expected
    assert row["sequence_embedding_path"] == str(tmp_path / f"{expected}.npz")


def test_manifest_naming_api_has_no_file_name_alias():
    assert not hasattr(manifest, "get_preprocessed_file_name")


def test_singleton_split_keeps_artifact_only_in_training():
    assert split_artifact_references(["artifact"]) == ([Path("artifact")], [])
    assert split_artifact_references([]) == ([], [])


def test_split_deduplicates_local_artifact_aliases_without_mutating_input(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(artifact, target_is_directory=True)
    paths = [
        Path("artifact"),
        artifact,
        alias,
        artifact / "metadata.json",
        Path("other"),
    ]
    original = paths.copy()
    train, valid = split_artifact_references(paths, validation_fraction=0.5, seed=3)
    assert paths == original
    assert train == [Path("artifact")]
    assert valid == [Path("other")]
    assert split_artifact_references(iter(paths), validation_fraction=0.5, seed=3) == (
        train,
        valid,
    )


def test_split_deduplicates_remote_roots_and_metadata_references():
    paths = [
        "s3://bucket/artifact",
        "s3://bucket/artifact/",
        "s3://bucket/artifact/metadata.json",
        "s3://bucket/other",
    ]
    train, valid = split_artifact_references(paths, validation_fraction=0.5, seed=3)
    assert train == ["s3://bucket/artifact"]
    assert valid == ["s3://bucket/other"]


def test_duplicate_only_split_remains_a_single_training_artifact():
    assert split_artifact_references(
        ["artifact", "artifact/metadata.json", "artifact"]
    ) == (
        [Path("artifact")],
        [],
    )


def test_bucket_root_artifact_split_deduplicates_metadata_reference():
    assert split_artifact_references(
        ["s3://bucket", "s3://bucket/", "s3://bucket/metadata.json"]
    ) == (["s3://bucket"], [])


def test_remote_split_keeps_significant_key_slashes():
    train, valid = split_artifact_references(
        ["s3://bucket/artifact", "s3://bucket//artifact"], seed=3
    )
    assert train == ["s3://bucket/artifact"]
    assert valid == ["s3://bucket//artifact"]


def test_split_validates_fraction_before_consuming_input():
    consumed = []

    def paths():
        consumed.append(True)
        yield "artifact"

    with pytest.raises(ValueError, match="validation_fraction"):
        split_artifact_references(paths(), validation_fraction=float("nan"))
    assert consumed == []


@pytest.mark.parametrize("suffix", [".prot", ".pkl", ".pickle", ".json", ""])
def test_manifest_rejects_unsupported_embedding_suffix_before_writing(tmp_path, suffix):
    output = tmp_path / "manifests" / "manifest.csv"
    with pytest.raises(ValueError, match="sequence_embedding_suffix"):
        write_manifest_csv(
            [ManifestEntry("1234", "1abc")],
            output,
            sequence_embeddings_dir=tmp_path / "embeddings",
            sequence_embedding_suffix=suffix,
        )
    assert not output.parent.exists()


def test_grouped_split_keeps_atomic_peers_and_replays_report(tmp_path):
    records = [
        GroupedSplitRecord("a", "map-a", ("protein-a",)),
        GroupedSplitRecord("a-cleaned", "map-a", ("protein-b",)),
        GroupedSplitRecord("b", "map-b", ("protein-b",)),
        GroupedSplitRecord("c", "map-c", ("protein-c",)),
    ]
    split = split_grouped_records(
        records,
        split_strategy="component",
        validation_fraction=0.5,
        seed=7,
        holdout_group_ids=["protein-a"],
    )
    assert split.assignments == ("valid", "valid", "train", "train")
    report = tmp_path / "split.json"
    write_grouped_split_report(split, report)
    replayed = split_grouped_records(
        records,
        split_strategy="component",
        validation_fraction=0.5,
        seed=7,
        holdout_group_ids=read_holdout_group_ids(
            report, expected_split_strategy="component"
        ),
    )
    assert replayed == split


@pytest.mark.parametrize(
    "payload", ['["protein-a", "protein-b"]', "protein-a\nprotein-b\n"]
)
def test_holdout_group_files_accept_simple_group_lists(tmp_path, payload):
    path = tmp_path / "holdout.txt"
    path.write_text(payload)
    assert read_holdout_group_ids(path) == ("protein-a", "protein-b")


def test_holdout_group_files_reject_unversioned_object_alias(tmp_path):
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps({"holdout_group_ids": ["protein-a"]}))
    with pytest.raises(ValueError, match="must be a split report"):
        read_holdout_group_ids(path)


@pytest.mark.parametrize("workers", [1, 3])
def test_downloads_skip_failed_maps_and_fetch_shared_structures_once(
    monkeypatch, tmp_path, workers
):
    entries = [
        ManifestEntry("1234", "1abc"),
        ManifestEntry("1235", "1abc"),
        ManifestEntry("1236", "2def"),
    ]
    calls = []
    lock = threading.Lock()

    def download(url, output_path, force_download=False):
        with lock:
            calls.append((url, output_path, force_download))
        if url == entries[2].map_url:
            raise HTTPError(url, 404, "missing", None, None)
        return output_path

    monkeypatch.setattr("cryoflake.manifests.download.download_url", download)
    with pytest.warns(RuntimeWarning, match="HTTP 404"):
        errors = download_manifest_entries(
            entries,
            maps_dir=tmp_path / "maps",
            structures_dir=tmp_path / "structures",
            force_download=True,
            num_workers=workers,
        )

    assert len(calls) == 4  # Three maps and one shared structure.
    assert all(force_download for _, _, force_download in calls)
    assert sum(url == entries[0].structure_url for url, _, _ in calls) == 1
    assert not any(url == entries[2].structure_url for url, _, _ in calls)
    assert set(errors) == {path_key(tmp_path / "maps" / entries[2].map_file_name)}
