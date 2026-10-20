"""The supported API uses one canonical name for each operation."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "removed_names"),
    [
        (
            "cryoflake.storage.codec",
            (
                "read_npz_json",
                "write_npz_json",
                "write_sequence_npz_json",
                "_read_metadata",
                "compressed_grid",
                "read_compressed_grid",
            ),
        ),
        ("cryoflake.storage.locks", ("preprocessed_write_lock",)),
        ("cryoflake.storage.transports", ("is_s3_uri",)),
        ("cryoflake.preprocess.manifest", ("set_default_preprocess_fn",)),
        (
            "cryoflake.sequence.sidecars",
            ("attach_sequence_embeddings_to_preprocessed_example",),
        ),
        (
            "cryoflake.preprocess.reuse",
            ("_preprocessed_example_is_reusable",),
        ),
        (
            "cryoflake.targets.full_atom",
            ("load_full_atom_targets_from_structure",),
        ),
        (
            "cryoflake.targets.components",
            ("validate_component_extension_targets",),
        ),
        ("cryoflake.io.fasta", ("FASTASequence",)),
    ],
)
def test_removed_forwarding_apis_are_absent(module_name, removed_names):
    module = import_module(module_name)
    for name in removed_names:
        assert not hasattr(module, name)


def test_public_convenience_exports_share_the_canonical_implementation():
    import cryoflake
    import cryoflake.storage as storage
    from cryoflake.sequence.validation import validate_sequence_embedding_payload
    from cryoflake.storage import store

    for name in storage.__all__:
        if hasattr(store, name):
            assert getattr(storage, name) is getattr(store, name)
            assert getattr(cryoflake, name) is getattr(store, name)
    assert (
        cryoflake.validate_sequence_embedding_payload
        is validate_sequence_embedding_payload
    )
