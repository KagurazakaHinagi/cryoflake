"""FASTA reader behavior independent of a particular biological alphabet."""

import pytest

from cryoflake.io.fasta import FASTAEmptyError, read_fasta


def test_fasta_preserves_duplicate_headers_and_normalizes_sequence_case(tmp_path):
    path = tmp_path / "sequences.fasta"
    path.write_text("; comment\n>same\nac d\nEF\n\n>same\nGG\n; ignored\nTT\n")
    records, names = read_fasta(path)
    assert names == ["same", "same"]
    assert [record.sequence for record in records] == ["ACDEF", "GGTT"]
    assert [record.chain_ids for record in records] == [["A"], ["A"]]


@pytest.mark.parametrize(
    "use_author_chain_ids,expected", [(True, ["X", "Y"]), (False, ["A", "B"])]
)
def test_fasta_rcsb_author_and_label_chain_mapping(
    tmp_path, use_author_chain_ids, expected
):
    path = tmp_path / "rcsb.fasta"
    path.write_text(">1ABC_1|Chains A [auth X], B [auth Y]|protein\nACDE\n")
    records, names = read_fasta(path, use_author_chain_ids=use_author_chain_ids)
    assert names == ["1ABC_1"]
    assert records[0].chain_ids == expected
    assert records[0].chain_count == 2


@pytest.mark.parametrize("contents", ["", "; comment\n", ">empty\n", "ACDE\n"])
def test_fasta_without_nonempty_records_raises(tmp_path, contents):
    path = tmp_path / "empty.fasta"
    path.write_text(contents)
    with pytest.raises(FASTAEmptyError, match="no nonempty FASTA records"):
        read_fasta(path)


def test_fasta_ignores_empty_records_and_preserves_valid_names(tmp_path):
    path = tmp_path / "mixed.fasta"
    path.write_text(">empty\n>valid|Chains |protein\nACGT\n")
    records, names = read_fasta(path)
    assert names == ["valid"]
    assert records[0].chain_ids == ["A"]
