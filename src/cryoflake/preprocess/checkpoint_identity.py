from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import types
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from cryoflake.common import sha256_file


@dataclass
class _IdentityCache:
    file_hashes: dict[Path, str] = field(default_factory=dict)
    source_trees: dict[Path, str] = field(default_factory=dict)
    module_versions: dict[str, str] = field(default_factory=dict)


_identity_cache: ContextVar[Optional[_IdentityCache]] = ContextVar(
    "cryoflake_callable_identity_cache", default=None
)
_ARRAY_HASH_CHUNK_BYTES = 1024**2


def _array_data_sha256(value: np.ndarray) -> str:
    """Hash C-order bytes with bounded copies for noncontiguous plain arrays."""

    digest = hashlib.sha256()
    if value.nbytes == 0:
        return digest.hexdigest()
    if value.flags.c_contiguous:
        digest.update(memoryview(value).cast("B"))
    elif value.dtype.fields is not None:
        # Structured dtype copies can treat padding bytes differently during
        # buffered iteration. Retain NumPy's existing copy semantics for these.
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    else:
        with np.nditer(
            value,
            flags=["external_loop", "buffered"],
            op_flags=["readonly"],
            order="C",
            buffersize=max(1, _ARRAY_HASH_CHUNK_BYTES // value.dtype.itemsize),
        ) as iterator:
            for chunk in iterator:
                digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest()


def _module_version(module_name: str) -> str:
    if not module_name:
        return ""
    top_level_name = module_name.split(".", 1)[0]
    cache = _identity_cache.get()
    if cache is not None and top_level_name in cache.module_versions:
        return cache.module_versions[top_level_name]
    try:
        version = importlib.metadata.version(top_level_name)
    except importlib.metadata.PackageNotFoundError:
        version = ""
    if cache is not None:
        cache.module_versions[top_level_name] = version
    return version


def _source_file_sha256(path: Path) -> str:
    cache = _identity_cache.get()
    if cache is None:
        return sha256_file(path)
    if path not in cache.file_hashes:
        cache.file_hashes[path] = sha256_file(path)
    return cache.file_hashes[path]


def _source_tree_sha256(
    source_path: Path,
    module_name: str,
) -> str:
    source_path = source_path.resolve()
    top_level_name = module_name.split(".", 1)[0]
    package_root = next(
        (
            candidate
            for candidate in (source_path.parent, *source_path.parents)
            if candidate.name == top_level_name
        ),
        None,
    )
    if (
        package_root is None
        or "site-packages" in package_root.parts
        or "dist-packages" in package_root.parts
    ):
        relative_root = source_path.parent
        tree_key = source_path
    else:
        relative_root = package_root
        tree_key = package_root

    cache = _identity_cache.get()
    if cache is not None and tree_key in cache.source_trees:
        return cache.source_trees[tree_key]
    source_paths = (
        [source_path] if tree_key == source_path else sorted(package_root.rglob("*.py"))
    )

    digest = hashlib.sha256()
    for candidate in source_paths:
        try:
            relative_path = candidate.relative_to(relative_root)
            source_digest = bytes.fromhex(_source_file_sha256(candidate))
        except OSError:
            continue
        digest.update(str(relative_path).encode())
        digest.update(b"\0")
        digest.update(source_digest)
    result = digest.hexdigest()
    if cache is not None:
        cache.source_trees[tree_key] = result
    return result


def _code_constant_identity(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"type": "float", "value": repr(value)}
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": repr(value.real),
            "imag": repr(value.imag),
        }
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if isinstance(value, types.CodeType):
        return {"type": "code", "value": _code_identity(value)}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_code_constant_identity(item) for item in value],
        }
    if isinstance(value, frozenset):
        items = [_code_constant_identity(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {"type": "frozenset", "items": items}
    return {
        "type": (f"{value.__class__.__module__}.{value.__class__.__qualname__}"),
        "repr": repr(value),
    }


def _code_identity(code: types.CodeType) -> dict[str, object]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "exceptiontable_sha256": hashlib.sha256(code.co_exceptiontable).hexdigest(),
        "constants": [_code_constant_identity(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _callable_reference_identity(value: Callable) -> dict[str, object]:
    instance_name = getattr(value, "__name__", "")
    source_target = getattr(value, "__func__", value)
    if not (
        inspect.isfunction(source_target)
        or inspect.ismethod(source_target)
        or inspect.isbuiltin(source_target)
        or inspect.isclass(source_target)
    ):
        source_target = value.__class__
    module_name = getattr(
        source_target,
        "__module__",
        value.__class__.__module__,
    ) or getattr(value.__class__, "__module__", "")
    qualified_name = (
        getattr(
            source_target,
            "__qualname__",
            value.__class__.__qualname__,
        )
        or value.__class__.__qualname__
    )
    identity: dict[str, object] = {
        "name": f"{module_name}.{qualified_name}",
        "package_version": _module_version(module_name),
        "instance_name": str(instance_name),
        "code_sha256": "",
        "source_sha256": "",
        "source_tree_sha256": "",
    }
    code = getattr(source_target, "__code__", None)
    if isinstance(code, types.CodeType):
        identity["code_sha256"] = hashlib.sha256(
            json.dumps(
                _code_identity(code),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    try:
        source_path_value = inspect.getsourcefile(source_target)
    except (OSError, TypeError):
        source_path_value = None
    if source_path_value:
        source_path = Path(source_path_value)
        if source_path.is_file():
            identity["source_sha256"] = _source_file_sha256(source_path.resolve())
            identity["source_tree_sha256"] = _source_tree_sha256(
                source_path,
                module_name,
            )
    return identity


def _identity_value(
    value: object,
    seen: Optional[set[int]] = None,
    *,
    include_callable_globals: bool = False,
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"type": "float", "value": repr(value)}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": repr(value.real),
            "imag": repr(value.imag),
        }
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if value is NotImplemented:
        return {"type": "not_implemented"}
    if isinstance(value, np.ma.MaskedArray):
        return {
            "type": "numpy.ma.MaskedArray",
            "data": _identity_value(
                np.asarray(value.data),
                seen,
                include_callable_globals=include_callable_globals,
            ),
            "mask": _identity_value(
                np.ma.getmaskarray(value),
                seen,
                include_callable_globals=include_callable_globals,
            ),
            "fill_value": _identity_value(
                value.fill_value,
                seen,
                include_callable_globals=include_callable_globals,
            ),
            "hard_mask": bool(value.hardmask),
            "shared_mask": bool(getattr(value, "_sharedmask", False)),
        }
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError(
                "Cannot safely fingerprint an object-dtype NumPy array; "
                "pass preprocess_checkpoint_token for this pipeline"
            )
        return {
            "type": "numpy.ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "strides": list(value.strides),
            "writeable": bool(value.flags.writeable),
            "sha256": _array_data_sha256(value),
        }
    if isinstance(value, np.ndarray):
        raise TypeError(
            "Cannot safely fingerprint NumPy array subclass "
            f"{value.__class__.__module__}.{value.__class__.__qualname__}; "
            "pass preprocess_checkpoint_token for this pipeline"
        )
    if isinstance(value, np.generic):
        return {
            "type": "numpy.scalar",
            "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.dtype):
        return {
            "type": "numpy.dtype",
            "value": value.str,
            "fields": _identity_value(
                value.descr,
                seen,
                include_callable_globals=include_callable_globals,
            ),
        }
    if isinstance(value, re.Pattern):
        return {
            "type": "regex",
            "pattern": _identity_value(
                value.pattern,
                seen,
                include_callable_globals=include_callable_globals,
            ),
            "flags": value.flags,
        }
    if isinstance(value, os.PathLike):
        return {"type": "path", "value": os.fspath(value)}
    if isinstance(value, types.ModuleType):
        return {
            "type": "module",
            "name": value.__name__,
            "version": str(getattr(value, "__version__", ""))
            or _module_version(value.__name__),
        }
    if callable(value):
        return {
            "type": "callable",
            "identity": _callable_identity_data(
                value,
                include_globals=include_callable_globals,
                seen=seen,
            ),
        }

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return {
            "type": (f"{value.__class__.__module__}.{value.__class__.__qualname__}"),
            "cycle": True,
        }
    seen.add(value_id)
    try:
        if isinstance(value, Mapping):
            items = [
                (
                    _identity_value(
                        key,
                        seen,
                        include_callable_globals=include_callable_globals,
                    ),
                    _identity_value(
                        item_value,
                        seen,
                        include_callable_globals=include_callable_globals,
                    ),
                )
                for key, item_value in value.items()
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item[0],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {"type": "mapping", "items": items}
        if isinstance(value, (list, tuple)):
            return {
                "type": value.__class__.__name__,
                "items": [
                    _identity_value(
                        item,
                        seen,
                        include_callable_globals=include_callable_globals,
                    )
                    for item in value
                ],
            }
        if isinstance(value, (set, frozenset)):
            items = [
                _identity_value(
                    item,
                    seen,
                    include_callable_globals=include_callable_globals,
                )
                for item in value
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {"type": value.__class__.__name__, "items": items}
        try:
            object_state = vars(value)
        except TypeError:
            object_state = None
        if not object_state:
            slots = getattr(value.__class__, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            object_state = {
                slot: getattr(value, slot) for slot in slots if hasattr(value, slot)
            }
        if object_state:
            return {
                "type": (
                    f"{value.__class__.__module__}.{value.__class__.__qualname__}"
                ),
                "state": _identity_value(
                    object_state,
                    seen,
                    include_callable_globals=include_callable_globals,
                ),
            }
        if getattr(value.__class__, "__repr__", None) is object.__repr__:
            return {
                "type": (
                    f"{value.__class__.__module__}.{value.__class__.__qualname__}"
                ),
                "opaque": True,
            }
        repr_value = repr(value)
        address_match = re.fullmatch(
            r"<(?P<label>[^<>]+) at 0x[0-9a-fA-F]+>",
            repr_value,
        )
        if address_match is not None:
            repr_value = f"<{address_match.group('label')} at 0x<address>>"
        elif "..." in repr_value:
            raise TypeError(
                "Cannot safely fingerprint truncated runtime state of "
                f"{value.__class__.__module__}."
                f"{value.__class__.__qualname__}; pass "
                "preprocess_checkpoint_token for this pipeline"
            )
        return {
            "type": (f"{value.__class__.__module__}.{value.__class__.__qualname__}"),
            "repr": repr_value,
        }
    finally:
        seen.remove(value_id)


def _identity_value_sha256(
    value: object,
    seen: Optional[set[int]] = None,
    *,
    include_callable_globals: bool = False,
) -> str:
    payload = json.dumps(
        _identity_value(
            value,
            seen,
            include_callable_globals=include_callable_globals,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _callable_identity_data(
    value: Callable,
    *,
    state_token: Optional[str] = None,
    include_globals: bool = True,
    seen: Optional[set[int]] = None,
) -> dict[str, object]:
    if not callable(value):
        raise TypeError("Callable identity requires a callable value")
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return {
            "kind": "callable_reference",
            **_callable_reference_identity(value),
            "cycle": True,
        }
    seen.add(value_id)
    try:
        if state_token is not None:
            if not isinstance(state_token, str):
                raise TypeError("Callable state token must be a string")
            reference_value = (
                value.func if isinstance(value, functools.partial) else value
            )
            return {
                "kind": "callable_with_explicit_state",
                **_callable_reference_identity(reference_value),
                "state_token_sha256": hashlib.sha256(state_token.encode()).hexdigest(),
            }

        if isinstance(value, functools.partial):
            return {
                "kind": "partial",
                "callable": _callable_identity_data(
                    value.func,
                    include_globals=include_globals,
                    seen=seen,
                ),
                "bound_state_sha256": _identity_value_sha256(
                    {
                        "args": value.args,
                        "keywords": value.keywords or {},
                    },
                    seen,
                    include_callable_globals=include_globals,
                ),
            }

        identity = {
            "kind": "callable",
            **_callable_reference_identity(value),
        }
        state: dict[str, object] = {}
        if inspect.isfunction(value):
            state["defaults"] = value.__defaults__
            state["keyword_defaults"] = value.__kwdefaults__
            closure_values = []
            for closure_cell in value.__closure__ or ():
                try:
                    closure_values.append(closure_cell.cell_contents)
                except ValueError:
                    closure_values.append({"empty_cell": True})
            state["closure"] = closure_values
            if include_globals:
                state["globals"] = {
                    name: value.__globals__[name]
                    for name in sorted(set(value.__code__.co_names))
                    if name in value.__globals__ and name != "__builtins__"
                }
        elif inspect.ismethod(value):
            state["function"] = _callable_identity_data(
                value.__func__,
                include_globals=include_globals,
                seen=seen,
            )
            state["bound_self"] = value.__self__
        elif inspect.isclass(value) or isinstance(value, np.ufunc):
            pass
        elif inspect.isbuiltin(value):
            bound_self = getattr(value, "__self__", None)
            if bound_self is not None and not isinstance(bound_self, types.ModuleType):
                state["bound_self"] = bound_self
        else:
            call_implementation = getattr(value.__class__, "__call__", None)
            if callable(call_implementation):
                state["call_implementation"] = _callable_identity_data(
                    call_implementation,
                    include_globals=include_globals,
                    seen=seen,
                )
            try:
                state["object_state"] = vars(value)
            except TypeError:
                slots = getattr(value.__class__, "__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                state["object_state"] = {
                    slot: getattr(value, slot) for slot in slots if hasattr(value, slot)
                }
        if state:
            identity["state_sha256"] = _identity_value_sha256(
                state,
                seen,
                include_callable_globals=include_globals,
            )
        return identity
    finally:
        seen.remove(value_id)


def callable_identity(
    value: Callable,
    *,
    state_token: Optional[str] = None,
) -> str:
    # Referenced functions often share a source package. Hash it once for this
    # snapshot, while re-reading it on later calls so edits invalidate recovery.
    token = _identity_cache.set(_IdentityCache())
    try:
        identity = _callable_identity_data(
            value,
            state_token=state_token,
        )
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))
    finally:
        _identity_cache.reset(token)
