"""Portable value types represented by the artifact codec."""

from __future__ import annotations

from typing import NamedTuple


class FASTASequence(NamedTuple):
    """One FASTA sequence and the chains represented by it."""

    sequence: str
    chain_count: int
    chain_ids: list[str]
