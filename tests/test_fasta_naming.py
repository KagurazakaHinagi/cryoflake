"""FASTA values have explicit fields without special handling of old schemas."""

from __future__ import annotations

import inspect
import json
from collections import namedtuple

import pytest

from cryoflake import FASTASequence, read_fasta
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)


def test_fasta_value_uses_sequence_and_chain_fields_without_aliases():
    value = FASTASequence(sequence="ACD", chain_count=2, chain_ids=["A", "B"])

    assert value._fields == ("sequence", "chain_count", "chain_ids")
    assert value.sequence == "ACD"
    assert value.chain_count == len(value.chain_ids) == 2
    for name in ("seq", "rep", "chains"):
        assert not hasattr(value, name)
    with pytest.raises(TypeError, match="seq"):
        FASTASequence(seq="ACD", rep=2, chains=["A", "B"])


def test_fasta_reader_uses_canonical_local_path_and_author_chain_keywords(tmp_path):
    fasta_path = tmp_path / "chains.fasta"
    fasta_path.write_text(">1ABC_1|Chains A [auth X], B [auth Y]|protein\nac d\n")

    records, names = read_fasta(fasta_path=fasta_path, use_author_chain_ids=True)

    assert names == ["1ABC_1"]
    assert records == [FASTASequence("ACD", 2, ["X", "Y"])]
    label_records, _ = read_fasta(fasta_path=fasta_path, use_author_chain_ids=False)
    assert label_records == [FASTASequence("ACD", 2, ["A", "B"])]
    assert list(inspect.signature(read_fasta).parameters) == [
        "fasta_path",
        "use_author_chain_ids",
    ]
    with pytest.raises(TypeError, match="auth_chains"):
        read_fasta(fasta_path, auth_chains=True)
    with pytest.raises(TypeError, match="path"):
        read_fasta(path=fasta_path)


def test_fasta_namedtuple_roundtrip_retains_only_the_canonical_recognized_schema(
    tmp_path,
):
    artifact_dir = tmp_path / "artifact"
    sequence = FASTASequence("ACD", 2, ["A", "B"])
    write_preprocessed_artifact(artifact_dir, {"fasta": sequence})

    restored = load_preprocessed_artifact(artifact_dir)["fasta"]
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    assert metadata["format_version"] == 1
    assert set(metadata["tree"]["items"]["fasta"]["items"]) == {
        "sequence",
        "chain_count",
        "chain_ids",
    }
    assert type(restored) is FASTASequence
    assert restored == sequence


@pytest.mark.parametrize(
    "type_name, fields, values",
    [
        ("FASTASequence", ("seq", "rep", "chains"), ("ACD", 2, ["A", "B"])),
        ("CustomSequence", ("sequence", "chain_count", "chain_ids"), ("ACD", 1, ["A"])),
        ("Coordinates", ("x", "y"), (1, 2)),
    ],
)
def test_unrecognized_namedtuples_keep_the_generic_field_mapping_fallback(
    tmp_path, type_name, fields, values
):
    value = namedtuple(type_name, fields)(*values)
    artifact_dir = tmp_path / "artifact"
    write_preprocessed_artifact(artifact_dir, {"value": value})

    restored = load_preprocessed_artifact(artifact_dir)["value"]

    assert type(restored) is dict
    assert restored == dict(zip(fields, values))
