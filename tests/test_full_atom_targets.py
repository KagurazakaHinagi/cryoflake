from types import SimpleNamespace

import numpy as np
import pytest
from standard_target import load_standard_full_atom_targets, standard_expected

from cryoflake.constants import (
    FULL_ATOM_COUNT,
    FULL_ATOM_TARGET_KEYS,
    FULL_ATOM_TARGET_SCHEMA_VERSION,
    RIGID_GROUP_COUNT,
    TORSION_COUNT,
    UNKNOWN_PROTEIN_INDEX,
)
from cryoflake.targets import common as target_common
from cryoflake.targets import full_atom
from cryoflake.targets.full_atom import (
    extract_full_atom_targets,
    validate_full_atom_targets,
)


def _target_source():
    return SimpleNamespace(
        **{
            key: value
            for key, value in load_standard_full_atom_targets().items()
            if key != "schema_version"
        }
    )


def _source_rows(source, rows):
    return SimpleNamespace(
        **{
            key: value if key == "atom_names" else value[rows]
            for key, value in vars(source).items()
        }
    )


def test_full_atom_targets_follow_ca_then_p_node_order():
    expected = load_standard_full_atom_targets()
    metadata = standard_expected()
    protein_count = metadata["protein_nodes"]
    residue_count = metadata["full_atom_residue_count"]
    # Deliberately put the authentic nucleotide block first. Extraction must
    # reorder every residue-indexed cache without touching the shared table.
    source_order = np.r_[protein_count:residue_count, :protein_count]
    targets = extract_full_atom_targets(_source_rows(_target_source(), source_order))

    for key, value in expected.items():
        np.testing.assert_array_equal(targets[key], value)
    assert "prot_mask" not in targets
    assert "prot_mask" not in FULL_ATOM_TARGET_KEYS
    protein_mask = (targets["aatype"] < 20) | (
        targets["aatype"] == UNKNOWN_PROTEIN_INDEX
    )
    np.testing.assert_array_equal(
        protein_mask, np.arange(residue_count) < protein_count
    )
    assert targets["atom_names"][:5].tolist() == ["N", "CA", "C", "O", "CB"]
    assert targets["atom_names"][[36, 37, 63, 64]].tolist() == [
        "OP1",
        "P",
        "O2'",
        "OXT",
    ]
    assert targets["atom_names"].shape == (FULL_ATOM_COUNT,)
    assert targets["atom_names"].dtype.kind == "U"
    assert targets["schema_version"] == FULL_ATOM_TARGET_SCHEMA_VERSION == "F2610210"
    assert targets["atom_positions"].dtype == np.float32
    assert targets["atom_mask"].dtype == np.bool_
    assert int(targets["atom_mask"].sum()) == metadata["present_atom_count"]
    assert not targets["atom_mask"].all()
    assert targets["rigidgroups_gt_frames"].shape == (
        residue_count,
        RIGID_GROUP_COUNT,
        3,
        4,
    )
    assert targets["torsion_angles_sin_cos"].shape == (residue_count, TORSION_COUNT, 2)
    assert np.any(np.abs(targets["rigidgroups_gt_frames"][..., 3]) > 1)
    assert np.any(np.abs(targets["torsion_angles_sin_cos"]) > 0.1)
    assert targets["torsion_angles_mask"].any()
    assert np.any(targets["b_factors"][targets["atom_mask"]] > 0)
    for sample_name in ("protein_ca_sample", "nucleotide_p_sample"):
        sample = metadata[sample_name]
        row, slot = sample["row"], sample["slot"]
        assert targets["atom_mask"][row, slot]
        assert targets["atom_names"][slot] == sample["atom_name"]
        assert targets["residue_index"][row] == sample["residue_index"]
        np.testing.assert_array_equal(
            targets["atom_positions"][row, slot],
            np.asarray(sample["coordinate"], dtype=np.float32),
        )
    frame = metadata["rigid_frame_sample"]
    np.testing.assert_array_equal(
        targets["rigidgroups_gt_frames"][frame["row"], frame["group"]],
        np.asarray(frame["matrix"], dtype=np.float32),
    )
    torsion = metadata["torsion_sample"]
    np.testing.assert_array_equal(
        targets["torsion_angles_sin_cos"][torsion["row"], torsion["torsion"]],
        np.asarray(torsion["sin_cos"], dtype=np.float32),
    )


def test_validate_full_atom_targets_checks_coordinate_counts():
    targets = extract_full_atom_targets(_target_source())
    protein_count = standard_expected()["protein_nodes"]
    nucleotide_count = standard_expected()["nucleotide_nodes"]

    with pytest.raises(ValueError, match="protein count"):
        validate_full_atom_targets(
            targets, num_protein=protein_count + 1, num_nucleotide=nucleotide_count
        )
    with pytest.raises(ValueError, match="nucleotide count"):
        validate_full_atom_targets(
            targets, num_protein=protein_count, num_nucleotide=nucleotide_count + 1
        )


@pytest.mark.parametrize("protein_mask", ["derived", None, "all_false"])
def test_full_atom_targets_reject_removed_protein_mask_even_when_consistent(
    protein_mask,
):
    targets = extract_full_atom_targets(_target_source())
    if protein_mask == "derived":
        protein_mask = (targets["aatype"] < 20) | (
            targets["aatype"] == UNKNOWN_PROTEIN_INDEX
        )
    elif protein_mask == "all_false":
        protein_mask = np.zeros(len(targets["aatype"]), dtype=bool)
    targets["prot_mask"] = protein_mask
    with pytest.raises(ValueError, match="prot_mask.*must not be stored"):
        validate_full_atom_targets(targets)


def test_protein_mask_derivation_preserves_unknown_proteins_and_node_order():
    source = _target_source()
    first_nucleotide = standard_expected()["protein_nodes"]
    selected_rows = [first_nucleotide, 0, 1]
    source = _source_rows(source, selected_rows)
    source.aatype[1] = UNKNOWN_PROTEIN_INDEX
    targets = extract_full_atom_targets(source)
    np.testing.assert_array_equal(targets["aatype"], source.aatype[[1, 2, 0]])
    np.testing.assert_array_equal(
        targets["atom_positions"], source.atom_positions[[1, 2, 0]]
    )
    normalized = validate_full_atom_targets(targets, num_protein=2, num_nucleotide=1)
    assert "prot_mask" not in normalized


def test_full_atom_targets_reject_unsorted_residue_classes_without_stored_mask():
    targets = extract_full_atom_targets(_target_source())
    targets["aatype"] = np.roll(targets["aatype"], 1)
    with pytest.raises(ValueError, match="protein rows before nucleotide rows"):
        validate_full_atom_targets(targets)


def test_validate_full_atom_targets_rejects_incomplete_payload():
    with pytest.raises(ValueError, match="missing required"):
        validate_full_atom_targets({"aatype": np.asarray([0])})


def test_validate_full_atom_targets_rejects_non_boolean_masks():
    targets = extract_full_atom_targets(_target_source())
    targets["atom_mask"] = targets["atom_mask"].astype(np.float32)
    targets["atom_mask"][0, 0] = 0.5

    with pytest.raises(ValueError, match="boolean values"):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize(
    "field", ["atom_positions", "b_factors", "torsion_angles_sin_cos"]
)
def test_full_atom_targets_reject_values_that_overflow_float32(field):
    targets = extract_full_atom_targets(_target_source())
    targets[field] = targets[field].astype(np.float64)
    targets[field].flat[0] = 1e100

    with pytest.raises(ValueError, match="finite"):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize("value", [1e-100, 1 + 1e-10])
def test_full_atom_masks_are_validated_before_float32_rounding(value):
    targets = extract_full_atom_targets(_target_source())
    targets["atom_mask"] = targets["atom_mask"].astype(np.float64)
    targets["atom_mask"][0, 0] = value

    with pytest.raises(ValueError, match="boolean values"):
        validate_full_atom_targets(targets)


def test_full_atom_residue_indices_do_not_wrap_uint64():
    targets = extract_full_atom_targets(_target_source())
    targets["residue_index"] = np.full(len(targets["aatype"]), 2**63, dtype=np.uint64)
    with pytest.raises(ValueError, match="int64|representable"):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize(
    "version",
    [
        0,
        1,
        2,
        -1,
        True,
        False,
        1.0,
        "1",
        None,
        np.int32(1),
        np.int64(1),
        np.uint64(1),
        np.bool_(True),
        np.asarray(1),
        "",
        "f2610210",
        "E2610210",
        "F2610200",
        "F2610211",
        "F2610219",
        "F26102110",
        "F2610320",
        "F261021",
        "F202610210",
        " F2610210",
        "F2610210\n",
        b"F2610210",
        np.asarray("F2610210"),
        np.asarray(["F2610210"]),
        ["F2610210"],
    ],
)
def test_full_atom_schema_version_rejects_unsupported_values(version):
    targets = extract_full_atom_targets(_target_source())
    targets["schema_version"] = version
    with pytest.raises(ValueError, match="unsupported full-atom schema_version"):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize("version", ["F2610210", np.str_("F2610210")])
def test_full_atom_schema_version_accepts_string_scalars_and_normalizes_to_python(
    version,
):
    targets = extract_full_atom_targets(_target_source())
    targets["schema_version"] = version
    normalized = validate_full_atom_targets(targets)
    assert normalized["schema_version"] == "F2610210"
    assert type(normalized["schema_version"]) is str


def test_full_atom_schema_version_cannot_be_omitted():
    targets = extract_full_atom_targets(_target_source())
    targets.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        validate_full_atom_targets(targets)


def test_full_atom_names_cannot_be_omitted_or_replaced_by_a_legacy_name():
    targets = extract_full_atom_targets(_target_source())
    targets["atom_name"] = targets.pop("atom_names")
    with pytest.raises(ValueError, match="missing required target arrays"):
        validate_full_atom_targets(targets)


def test_extract_full_atom_names_requires_supplied_names_and_does_not_infer_slot_order():
    source = _target_source()
    del source.atom_names
    with pytest.raises(ValueError, match="missing atom_names"):
        extract_full_atom_targets(source)


@pytest.mark.parametrize("residue_count", [0, 1, 3, 65])
def test_extract_full_atom_names_rejects_per_residue_tables(
    residue_count,
):
    source = _target_source()
    source.atom_names = np.tile(source.atom_names, (residue_count, 1))
    with pytest.raises(ValueError, match="atom_names.*shape"):
        extract_full_atom_targets(source)


@pytest.mark.parametrize(
    "names",
    [
        np.full(64, "C"),
        np.full(66, "C"),
        np.full((65, 1), "C"),
        np.full((3, 65), "C"),
        "C",
        [["N"], ["N", "CA"], ["P"]],
    ],
)
def test_full_atom_names_reject_incorrect_and_ragged_shapes(names):
    targets = extract_full_atom_targets(_target_source())
    targets["atom_names"] = names
    with pytest.raises(ValueError, match="shape"):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize("invalid_name", [None, 1, 1.5, True, object()])
def test_full_atom_names_reject_nonstring_values_without_coercion(invalid_name):
    targets = extract_full_atom_targets(_target_source())
    names = targets["atom_names"].tolist()
    names[0] = invalid_name
    targets["atom_names"] = names
    with pytest.raises(ValueError, match="only strings"):
        validate_full_atom_targets(targets)


def test_extract_full_atom_names_preserves_mixed_sequence_types_until_validation():
    source = _target_source()
    source.atom_names = source.atom_names.tolist()
    source.atom_names[0] = 123
    with pytest.raises(ValueError, match="only strings"):
        extract_full_atom_targets(source)


@pytest.mark.parametrize(
    "invalid_name", ["", "  ", "bad name", "C\nA", "N\x00", ";N", "N/CA", "N@1"]
)
def test_full_atom_names_reject_missing_or_invalid_slot_names(invalid_name):
    targets = extract_full_atom_targets(_target_source())
    targets["atom_names"] = targets["atom_names"].astype(object)
    targets["atom_names"][0] = invalid_name
    with pytest.raises(ValueError, match="nonempty names|invalid atom name"):
        validate_full_atom_targets(targets)


def test_full_atom_names_reject_duplicates_after_normalization():
    targets = extract_full_atom_targets(_target_source())
    targets["atom_names"] = targets["atom_names"].astype(object)
    targets["atom_names"][:2] = ["ca", " CA "]
    with pytest.raises(ValueError, match="duplicate atom names"):
        validate_full_atom_targets(targets)


def test_full_atom_masked_slots_keep_names_without_zero_padding_requirements():
    targets = extract_full_atom_targets(_target_source())
    expected_names = targets["atom_names"].copy()
    targets["atom_mask"][:, 1:] = False
    targets["atom_positions"][:, 1:] = 23
    targets["b_factors"][:, 1:] = 17
    normalized = validate_full_atom_targets(targets)
    np.testing.assert_array_equal(normalized["atom_names"], expected_names)
    np.testing.assert_array_equal(normalized["atom_mask"], targets["atom_mask"])
    np.testing.assert_array_equal(normalized["atom_positions"][:, 1:], 23)
    np.testing.assert_array_equal(normalized["b_factors"][:, 1:], 17)


@pytest.mark.parametrize("invalid_name", ["", "  ", " ca ", "C;injected"])
def test_full_atom_names_validate_slots_even_when_unobserved_in_every_residue(
    invalid_name,
):
    targets = extract_full_atom_targets(_target_source())
    targets["atom_mask"][:, 0] = False
    targets["atom_names"] = targets["atom_names"].astype(object)
    targets["atom_names"][0] = invalid_name
    with pytest.raises(
        ValueError, match="nonempty names|duplicate atom names|invalid atom name"
    ):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize("array_dtype", [object, "S16"])
def test_full_atom_names_decode_utf8_bytes_and_normalize_to_unicode(array_dtype):
    targets = extract_full_atom_targets(_target_source())
    expected_names = targets["atom_names"].copy()
    names = np.asarray(
        [f" {name.lower()} ".encode("utf-8") for name in expected_names],
        dtype=array_dtype,
    )
    targets["atom_names"] = names
    normalized = validate_full_atom_targets(targets)
    assert normalized["atom_names"].dtype.kind == "U"
    np.testing.assert_array_equal(normalized["atom_names"], expected_names)


def test_full_atom_names_reject_invalid_utf8_bytes():
    targets = extract_full_atom_targets(_target_source())
    targets["atom_names"] = targets["atom_names"].astype(object)
    targets["atom_names"][0] = b"\xff"
    with pytest.raises(ValueError, match="valid UTF-8 strings"):
        validate_full_atom_targets(targets)


def test_full_atom_names_do_not_truncate_long_identifiers():
    targets = extract_full_atom_targets(_target_source())
    targets["atom_names"] = targets["atom_names"].astype(object)
    long_name = "C_" + "1" * 90
    targets["atom_names"][0] = long_name
    normalized = validate_full_atom_targets(targets)
    assert normalized["atom_names"][0] == long_name


def test_empty_full_atom_residue_groups_preserve_all_tensor_shapes():
    source = _target_source()
    empty_source = SimpleNamespace(
        **{
            key: value if key == "atom_names" else value[:0]
            for key, value in vars(source).items()
        }
    )
    targets = extract_full_atom_targets(empty_source)
    normalized = validate_full_atom_targets(targets, num_protein=0, num_nucleotide=0)
    assert normalized["schema_version"] == "F2610210"
    assert normalized["atom_names"].shape == (65,)
    np.testing.assert_array_equal(
        normalized["atom_names"], np.char.upper(np.char.strip(source.atom_names))
    )
    assert normalized["atom_names"].dtype.kind == "U"
    assert normalized["atom_mask"].shape == (0, 65)
    assert normalized["atom_positions"].shape == (0, 65, 3)


def test_empty_numeric_full_atom_names_are_rejected_as_nonstring_input():
    source = _target_source()
    empty_source = SimpleNamespace(
        **{
            key: value if key == "atom_names" else value[:0]
            for key, value in vars(source).items()
        }
    )
    empty_source.atom_names = np.zeros(65, dtype=float)
    with pytest.raises(ValueError, match="only strings"):
        extract_full_atom_targets(empty_source)


@pytest.mark.parametrize(
    "field",
    [
        key
        for key in (*FULL_ATOM_TARGET_KEYS, "chain_index", "residue_index")
        if key not in {"aatype", "atom_names"}
    ],
)
@pytest.mark.parametrize("row_difference", [-1, 1])
def test_extract_full_atom_targets_rejects_unaligned_source_rows(field, row_difference):
    source = _target_source()
    values = getattr(source, field)
    if row_difference < 0:
        values = values[:-1]
    else:
        values = np.concatenate((values, values[:1]), axis=0)
    setattr(source, field, values)

    with pytest.raises(ValueError, match=f"missing aligned {field} targets"):
        extract_full_atom_targets(source)


@pytest.mark.parametrize("aatype", [None, [0.0, 20.0, 24.0], [0, "20", 24]])
def test_extract_full_atom_targets_rejects_noninteger_source_types(aatype):
    source = _target_source()
    source.aatype = aatype
    with pytest.raises(ValueError, match="aatype.*one-dimensional integer"):
        extract_full_atom_targets(source)


def test_full_atom_float_validation_checks_normalized_arrays_once(monkeypatch):
    targets = extract_full_atom_targets(_target_source())
    checked = []
    all_finite = full_atom.array_all_finite

    def record_finite(array):
        checked.append(array)
        return all_finite(array)

    monkeypatch.setattr(full_atom, "array_all_finite", record_finite)
    normalized = validate_full_atom_targets(targets)
    floating_fields = (
        "atom_positions",
        "b_factors",
        "rigidgroups_gt_frames",
        "rigidgroups_alt_gt_frames",
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
    )
    assert len(checked) == len(floating_fields)
    assert {id(array) for array in checked} == {
        id(normalized[field]) for field in floating_fields
    }
    for field in floating_fields:
        assert normalized[field] is targets[field]
        np.testing.assert_array_equal(normalized[field], targets[field])


def test_numeric_full_atom_masks_use_bounded_checks_and_preserve_strided_values(
    monkeypatch,
):
    targets = extract_full_atom_targets(_target_source())
    backing = np.zeros((len(targets["aatype"]), FULL_ATOM_COUNT * 2), dtype=np.float64)
    backing[:, ::2] = targets["atom_mask"]
    targets["atom_mask"] = backing[:, ::2]
    assert not targets["atom_mask"].flags.c_contiguous
    checked_sizes = []
    isin = target_common.np.isin

    def record_isin(values, choices):
        checked_sizes.append(values.size)
        return isin(values, choices)

    monkeypatch.setattr(target_common, "_MASK_CHECK_BUFFER_ELEMENTS", 17)
    monkeypatch.setattr(target_common.np, "isin", record_isin)
    normalized = validate_full_atom_targets(targets)
    np.testing.assert_array_equal(normalized["atom_mask"], backing[:, ::2])
    assert len(checked_sizes) > 1
    assert max(checked_sizes) <= 17

    targets["atom_mask"][-1, -1] = np.nan
    with pytest.raises(ValueError, match="boolean values"):
        validate_full_atom_targets(targets)


def test_boolean_full_atom_masks_need_no_numeric_membership_scan(monkeypatch):
    targets = extract_full_atom_targets(_target_source())

    def unexpected_isin(*args, **kwargs):
        raise AssertionError("boolean masks already contain only false/true")

    monkeypatch.setattr(target_common.np, "isin", unexpected_isin)
    normalized = validate_full_atom_targets(targets)
    for field in (
        "atom_mask",
        "rigidgroups_gt_exists",
        "rigidgroups_group_exists",
        "rigidgroups_group_is_ambiguous",
        "torsion_angles_mask",
    ):
        assert normalized[field] is targets[field]
