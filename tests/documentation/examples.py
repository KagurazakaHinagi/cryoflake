"""Read explicitly classified example blocks from the checked-in Markdown."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION = re.compile(r"<!-- doc-example: ([a-z][a-z0-9-]*); mode=([a-z]+) -->")
FENCE = re.compile(r"(`{3,}|~{3,})(.*)")
_CONTAINER_PREFIX = re.compile(r"^(?:>\s*|(?:[-+*]|\d+[.)])\s+)")
LANGUAGES = {
    "python": {"python"},
    "cli": {"bash"},
    "options": {"text"},
    "data": {"json", "csv", "yaml", "dotenv"},
    "setup": {"bash"},
    "reference": {"text"},
}


@dataclass(frozen=True)
class Example:
    identifier: str
    mode: str
    language: str
    code: str
    path: Path
    line: int

    @property
    def location(self) -> str:
        return f"{self.path.relative_to(PROJECT_ROOT)}:{self.line}"

    def execute(self, namespace: Mapping[str, object] | None = None) -> dict:
        if self.language != "python":
            raise ValueError(f"{self.location}: expected a Python example")
        values = {"__name__": "__doc_example__", **(namespace or {})}
        source = "\n" * (self.line - 1) + self.code
        exec(compile(source, str(self.path), "exec"), values)
        return values

    @property
    def python_setup(self) -> str:
        """Extract Python heredocs without executing a shell."""
        return "\n".join(
            match.group(1)
            for match in re.finditer(
                r"^uv run python - <<'PY'\n(.*?)^PY\n?", self.code, re.M | re.S
            )
        )

    def shell_commands(self) -> list[list[str]]:
        """Tokenize simple documented commands; reject unsupported shell syntax."""
        source = re.sub(
            r"^uv run python - <<'PY'\n.*?^PY\n?", "", self.code, flags=re.M | re.S
        )
        source = source.replace("\\\n", " ")
        commands = []
        for line in source.splitlines():
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            if any(token in {";", "&&", "||", "|", ">", ">>"} for token in tokens):
                raise ValueError(f"{self.location}: unsupported shell composition")
            commands.append(tokens)
        return commands

    def commands(
        self, replacements: Mapping[str, str] | None = None
    ) -> list[list[str]]:
        """Return CLI arguments, skipping setup commands only when classified."""
        commands = []
        for tokens in self.shell_commands():
            tokens = [(replacements or {}).get(token, token) for token in tokens]
            while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
                tokens = tokens[1:]
            if tokens[:3] == ["uv", "run", "cryoflake"]:
                commands.append(tokens[3:])
            elif tokens[:1] == ["cryoflake"]:
                commands.append(tokens[1:])
            elif self.mode == "setup" and (
                tokens[:2] in (["uv", "sync"], ["uv", "build"], ["uv", "lock"])
                or tokens[:3]
                in (
                    ["uv", "pip", "install"],
                    ["uv", "run", "pytest"],
                    ["uv", "run", "ruff"],
                )
            ):
                continue
            else:
                raise ValueError(f"{self.location}: unhandled command {tokens!r}")
        return commands


def fence_closes(line: str, marker: str) -> bool:
    return len(line) >= len(marker) and set(line) == {marker[0]}


def documentation_paths() -> list[Path]:
    """Discover public Markdown guides, including nested and uppercase suffixes."""

    return [
        PROJECT_ROOT / "README.md",
        *sorted(
            path
            for path in (PROJECT_ROOT / "docs").rglob("*")
            if path.is_file() and path.suffix.lower() == ".md"
        ),
    ]


def parse_examples(path: Path) -> list[Example]:
    """Every fence needs an ID and an execution mode; omissions fail collection."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    examples = []
    pending = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        unwrapped = line
        while container := _CONTAINER_PREFIX.match(unwrapped):
            unwrapped = unwrapped[container.end() :].lstrip()
        if unwrapped != line and (
            FENCE.fullmatch(unwrapped) or ANNOTATION.fullmatch(unwrapped)
        ):
            raise ValueError(
                f"{path}:{index + 1}: documentation examples must use top-level fences"
            )
        annotation = ANNOTATION.fullmatch(line)
        if annotation:
            if pending is not None:
                raise ValueError(f"{path}:{index + 1}: unused example annotation")
            pending = annotation.groups()
        elif fence := FENCE.fullmatch(line):
            if pending is None:
                raise ValueError(
                    f"{path}:{index + 1}: code fence has no doc-example ID"
                )
            marker, language = fence.groups()
            language = language.strip()
            identifier, mode = pending
            if language not in LANGUAGES.get(mode, set()):
                raise ValueError(f"{path}:{index + 1}: invalid example mode/language")
            start = index + 1
            index = start
            while index < len(lines) and not fence_closes(lines[index].strip(), marker):
                index += 1
            if index == len(lines):
                raise ValueError(f"{path}:{start}: unterminated code fence")
            examples.append(
                Example(
                    identifier=identifier,
                    mode=mode,
                    language=language,
                    code="".join(lines[start:index]),
                    path=path,
                    line=start + 1,
                )
            )
            pending = None
        elif pending is not None and line:
            raise ValueError(f"{path}:{index + 1}: annotation must precede its fence")
        index += 1
    if pending is not None:
        raise ValueError(f"{path}: unused example annotation")
    return examples


def load_examples() -> dict[str, Example]:
    examples = {}
    for path in documentation_paths():
        for example in parse_examples(path):
            if example.identifier in examples:
                raise ValueError(f"{example.location}: duplicate example ID")
            examples[example.identifier] = example
    return examples


def get_example(identifier: str) -> Example:
    # Read on demand so a failing or edited example is never hidden by a cache.
    return load_examples()[identifier]
