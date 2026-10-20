"""Process-local serialization for writes to one preprocessed artifact root."""

from __future__ import annotations

import os
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from cryoflake.storage.transports import is_remote_uri

_LOCKS_GUARD = threading.Lock()


@dataclass
class _LockEntry:
    lock: threading.RLock = field(default_factory=threading.RLock)
    users: int = 0


_ROOT_LOCKS: dict[str, _LockEntry] = {}


def _lock_key(artifact_reference: os.PathLike | str) -> str:
    value = os.fspath(artifact_reference)
    if is_remote_uri(value):
        return value.rstrip("/")
    return str(Path(value).resolve())


@contextmanager
def preprocessed_artifact_write_locks(*artifact_references: os.PathLike | str):
    """Serialize writes to artifact roots, acquiring locks in a stable order."""

    keys = sorted(
        {_lock_key(artifact_reference) for artifact_reference in artifact_references}
    )
    with _LOCKS_GUARD:
        entries = []
        for key in keys:
            entry = _ROOT_LOCKS.setdefault(key, _LockEntry())
            entry.users += 1
            entries.append((key, entry))
    try:
        with ExitStack() as stack:
            for _, entry in entries:
                stack.enter_context(entry.lock)
            yield
    finally:
        with _LOCKS_GUARD:
            for key, entry in entries:
                entry.users -= 1
                if entry.users == 0 and _ROOT_LOCKS.get(key) is entry:
                    del _ROOT_LOCKS[key]
