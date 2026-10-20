from types import SimpleNamespace

import numpy as np
import pytest

from cryoflake.constants import (
    FULL_ATOM_COUNT,
    RIGID_GROUP_COUNT,
    TORSION_COUNT,
)
from cryoflake.targets.full_atom import (
    extract_full_atom_targets,
    validate_full_atom_targets,
)


def _target_source():
    num_residues = 3
    row_values = np.arange(num_residues, dtype=np.float32)

    def filled(trailing_shape):
        values = np.zeros((num_residues, *trailing_shape), dtype=np.float32)
        values[...] = row_values.reshape((num_residues,) + (1,) * len(trailing_shape))
        return values

    return SimpleNamespace(
        aatype=np.asarray([20, 0, 24], dtype=np.int64),
        chain_index=np.asarray([1, 0, 2], dtype=np.int64),
        residue_index=np.asarray([5, 10, 6], dtype=np.int64),
        atom_positions=filled((FULL_ATOM_COUNT, 3)),
        atom_mask=np.ones((num_residues, FULL_ATOM_COUNT), dtype=np.float32),
        b_factors=filled((FULL_ATOM_COUNT,)),
        rigidgroups_gt_frames=filled((RIGID_GROUP_COUNT, 3, 4)),
        rigidgroups_gt_exists=np.ones(
            (num_residues, RIGID_GROUP_COUNT), dtype=np.float32
        ),
        rigidgroups_group_exists=np.ones(
            (num_residues, RIGID_GROUP_COUNT), dtype=np.float32
        ),
        rigidgroups_group_is_ambiguous=np.zeros(
            (num_residues, RIGID_GROUP_COUNT), dtype=np.float32
        ),
        rigidgroups_alt_gt_frames=filled((RIGID_GROUP_COUNT, 3, 4)),
        torsion_angles_sin_cos=filled((TORSION_COUNT, 2)),
        alt_torsion_angles_sin_cos=filled((TORSION_COUNT, 2)),
        torsion_angles_mask=np.ones((num_residues, TORSION_COUNT), dtype=np.float32),
    )


def test_full_atom_targets_follow_ca_then_p_node_order():
    targets = extract_full_atom_targets(target_source=_target_source())

    np.testing.assert_array_equal(targets["aatype"], [0, 20, 24])
    np.testing.assert_array_equal(targets["prot_mask"], [True, False, False])
    np.testing.assert_array_equal(targets["chain_index"], [0, 1, 2])
    np.testing.assert_array_equal(targets["residue_index"], [10, 5, 6])
    np.testing.assert_array_equal(targets["atom_positions"][:, 0, 0], [1, 0, 2])
    assert targets["atom_positions"].dtype == np.float32
    assert targets["atom_mask"].dtype == np.bool_
    assert targets["rigidgroups_gt_frames"].shape == (3, 9, 3, 4)
    assert targets["torsion_angles_sin_cos"].shape == (3, 8, 2)


def test_validate_full_atom_targets_checks_coordinate_counts():
    targets = extract_full_atom_targets(_target_source())

    with pytest.raises(ValueError, match="protein count"):
        validate_full_atom_targets(targets, num_protein=2, num_nucleotide=1)


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
    targets["residue_index"] = np.full(3, 2**63, dtype=np.uint64)
    with pytest.raises(ValueError, match="int64|representable"):
        validate_full_atom_targets(targets)
