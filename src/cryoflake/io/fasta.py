"""FASTA parsing with Biotite, including RCSB author-chain headers."""

from __future__ import annotations

import re
from pathlib import Path

from biotite.sequence.io.fasta import FastaFile

from cryoflake import types


class FASTAEmptyError(RuntimeError):
    """Raised when a FASTA file contains no nonempty sequence records."""


def read_fasta(
    fasta_path: str | Path,
    use_author_chain_ids: bool = True,
) -> tuple[list[types.FASTASequence], list[str]]:
    """Read sequences and names, retaining duplicate records in file order.

    RCSB headers may map label chains to author chains, e.g. ``Chain A [auth X]``.
    Plain FASTA headers use chain ``A``. Sequence case is normalized to uppercase.
    """
    sequences = []
    sequence_names = []
    # Streaming avoids FastaFile's mapping interface collapsing duplicate headers.
    for header, sequence in FastaFile.read_iter(fasta_path):
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            continue
        description = header.split("|")
        sequence_names.append(description[0].strip() or "unknown")
        if len(description) > 1:
            chain_ids = []
            for chain in description[1].split(","):
                chain = re.sub(r"^\s*Chains?\s*", "", chain).strip()
                label_chain, _, author_chain = chain.partition("[auth")
                chain = (
                    author_chain.rstrip("]").strip()
                    if use_author_chain_ids and author_chain
                    else label_chain.strip()
                )
                if chain:
                    chain_ids.append(chain)
            chain_ids = chain_ids or ["A"]
        else:
            chain_ids = ["A"]
        sequences.append(types.FASTASequence(sequence, len(chain_ids), chain_ids))

    if not sequences:
        raise FASTAEmptyError(
            f"File {fasta_path!s} contains no nonempty FASTA records."
        )
    return sequences, sequence_names


__all__ = ["FASTAEmptyError", "read_fasta"]
