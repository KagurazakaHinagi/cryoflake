"""Keep every public fenced example classified and covered by the suite."""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

import cryoflake

from .examples import (
    FENCE,
    PROJECT_ROOT,
    Example,
    documentation_paths,
    fence_closes,
    load_examples,
    parse_examples,
)

pytestmark = pytest.mark.documentation

# This inventory is deliberately explicit. A new fence needs both a stable ID
# and a scenario here/in the example tests, rather than silently going untested.
EXPECTED_EXAMPLES = {
    "readme-install": "setup",
    "readme-manifest": "cli",
    "readme-inspect": "cli",
    "readme-storage": "python",
    "guide-install": "setup",
    "guide-wheel-install": "setup",
    "guide-manifest": "cli",
    "guide-download": "cli",
    "guide-single": "cli",
    "guide-batch": "cli",
    "guide-batch-python": "python",
    "guide-embeddings-options": "options",
    "guide-embeddings-python": "python",
    "guide-split-options": "options",
    "guide-group-options": "options",
    "guide-replay-options": "options",
    "guide-remote-options": "options",
    "guide-env": "data",
    "guide-env-file": "cli",
    "guide-validate": "cli",
    "config-inspect": "cli",
    "config-layout": "reference",
    "config-project": "cli",
    "config-lab": "data",
    "config-python": "python",
    "api-storage-signatures": "reference",
    "api-storage": "python",
    "api-fasta": "python",
    "api-validation-signatures": "reference",
    "artifact-layout": "reference",
    "artifact-references": "python",
    "artifact-tree": "data",
    "artifact-roundtrip": "python",
    "contract-source-csv": "data",
    "contract-geometry": "python",
    "contract-full-atom-roundtrip": "python",
    "contract-component-derived": "python",
    "development-checks": "setup",
}


def test_every_documented_fence_has_a_classified_regression_scenario():
    assert {
        identifier: example.mode for identifier, example in load_examples().items()
    } == EXPECTED_EXAMPLES


@pytest.mark.parametrize(
    "source, message",
    [
        ("```python\npass\n```\n", "no doc-example ID"),
        ("~~~python\npass\n~~~\n", "no doc-example ID"),
        (
            "<!-- doc-example: new; mode=python -->\n```bash\necho x\n```\n",
            "invalid example mode/language",
        ),
        (
            "<!-- doc-example: new; mode=python -->\n```python\npass\n",
            "unterminated code fence",
        ),
        (
            "<!-- doc-example: new; mode=python -->\n~~~~python\npass\n~~~\n",
            "unterminated code fence",
        ),
        ("<!-- doc-example: new; mode=python -->\n", "unused example annotation"),
    ],
)
def test_uncovered_or_misclassified_examples_fail_explicitly(tmp_path, source, message):
    document = tmp_path / "guide.md"
    document.write_text(source)
    with pytest.raises(ValueError, match=message):
        parse_examples(document)


@pytest.mark.parametrize("marker", ["```", "~~~", "````", "~~~~"])
def test_markdown_fence_styles_preserve_executable_source(tmp_path, marker):
    document = tmp_path / "guide.md"
    document.write_text(
        f"<!-- doc-example: new; mode=python -->\n{marker}python\n"
        f"result = 42\n{marker}{marker[0]}\n"
    )
    (example,) = parse_examples(document)
    assert example.execute()["result"] == 42
    assert example.line == 3


def test_nested_guides_are_included_in_the_example_inventory(tmp_path, monkeypatch):
    from . import examples

    (tmp_path / "README.md").write_text("")
    nested = tmp_path / "docs/tutorials/first-steps.md"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        "<!-- doc-example: nested; mode=python -->\n~~~python\nresult = 42\n~~~\n"
    )
    monkeypatch.setattr(examples, "PROJECT_ROOT", tmp_path)
    assert load_examples()["nested"].execute()["result"] == 42


@pytest.mark.parametrize("prefix", ["> ", "> > ", "- ", "1. ", "- > "])
def test_container_fences_cannot_bypass_example_inventory(tmp_path, prefix):
    document = tmp_path / "guide.md"
    document.write_text(f"{prefix}```python\n{prefix}result = 42\n{prefix}```\n")

    with pytest.raises(ValueError, match="top-level fences"):
        parse_examples(document)


def test_uppercase_markdown_guides_are_discovered(tmp_path, monkeypatch):
    from . import examples

    (tmp_path / "README.md").write_text("")
    guide = tmp_path / "docs/tutorials/Guide.MD"
    guide.parent.mkdir(parents=True)
    guide.write_text(
        "<!-- doc-example: uppercase; mode=python -->\n```python\nresult = 42\n```\n"
    )
    monkeypatch.setattr(examples, "PROJECT_ROOT", tmp_path)

    assert guide in documentation_paths()
    assert load_examples()["uppercase"].execute()["result"] == 42


def test_python_failures_point_to_the_markdown_source():
    example = Example(
        "broken",
        "python",
        "python",
        "raise ValueError('example failed')\n",
        PROJECT_ROOT / "README.md",
        42,
    )
    with pytest.raises(ValueError) as error:
        example.execute()
    assert error.value.__traceback__.tb_next.tb_next.tb_lineno == 42
    assert error.value.__traceback__.tb_next.tb_next.tb_frame.f_code.co_filename == str(
        PROJECT_ROOT / "README.md"
    )


@pytest.mark.parametrize(
    "identifier, names",
    [
        (
            "api-storage-signatures",
            {
                "load_preprocessed_artifact",
                "write_preprocessed_artifact",
                "replace_preprocessed_artifact_sequence",
                "upload_preprocessed_artifact",
            },
        ),
        (
            "api-validation-signatures",
            {"validate_preprocessed_artifact", "validate_artifact_list"},
        ),
    ],
)
def test_documented_signatures_match_the_public_functions(identifier, names):
    example = load_examples()[identifier]
    pattern = r"([a-z_]+)\((.*?)\)"
    signatures = re.findall(pattern, example.code, re.S)
    assert {name for name, _ in signatures} == names
    assert len(signatures) == len(names)
    assert not re.sub(pattern, "", example.code, flags=re.S).strip()
    for name, parameters in signatures:
        namespace = {}
        exec(f"def {name}({parameters}): pass", namespace)
        documented = inspect.signature(namespace[name])
        actual = inspect.signature(getattr(cryoflake, name))
        actual = actual.replace(
            parameters=[
                parameter.replace(annotation=inspect.Parameter.empty)
                for parameter in actual.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        assert documented == actual, f"{example.location}: {name}"


def _resolve_public_api(name: str):
    if name.startswith("cryoflake."):
        module, attribute = name.rsplit(".", 1)
        return getattr(importlib.import_module(module), attribute)
    modules = [
        cryoflake,
        *(
            importlib.import_module("cryoflake." + suffix)
            for suffix in (
                "config",
                "io.mrc",
                "manifests.manifest",
                "manifests.splits",
                "storage.upload_cache",
            )
        ),
    ]
    cache = modules[-1].BatchedArtifactUploadCache
    modules.append(cache)
    first, *attributes = name.split(".")
    candidates = {
        id(value): value
        for module in modules
        if (value := getattr(module, first, None)) is not None
    }
    assert len(candidates) == 1, f"Unresolved or ambiguous documented API: {name}"
    target = next(iter(candidates.values()))
    for attribute in attributes:
        target = getattr(target, attribute)
    return target


def _assert_inline_signature(name: str, parameters: str) -> None:
    # Inline synopses may omit optional trailing parameters or show defaults as ... .
    parameters = re.sub(r",\s*\.\.\.\s*$", "", parameters)
    namespace = {}
    exec(f"def documented({parameters}): pass", namespace)
    documented = inspect.signature(namespace["documented"]).parameters
    actual = dict(inspect.signature(_resolve_public_api(name)).parameters)
    actual.pop("self", None)
    assert set(documented).issubset(actual), f"{name}: invalid parameter names"
    assert list(documented) == [key for key in actual if key in documented], (
        f"{name}: incorrect parameter order"
    )
    required = {
        key
        for key, value in actual.items()
        if value.default is inspect.Parameter.empty
        and value.kind not in {value.VAR_POSITIONAL, value.VAR_KEYWORD}
    }
    assert required.issubset(documented), f"{name}: missing required parameters"
    for key, parameter in documented.items():
        assert parameter.kind == actual[key].kind, f"{name}: {key} parameter kind"
        assert (parameter.default is inspect.Parameter.empty) == (
            actual[key].default is inspect.Parameter.empty
        ), f"{name}: {key} required/optional status"
        if parameter.default not in (inspect.Parameter.empty, Ellipsis):
            assert parameter.default == actual[key].default, f"{name}: {key} default"


def test_inline_api_reference_signatures_match_runtime():
    source = (PROJECT_ROOT / "docs/api-reference.md").read_text()
    signatures = []
    for code in re.findall(r"(?<!`)`(?!`)([^`]+)`(?!`)", source, re.S):
        if match := re.fullmatch(r"([A-Za-z_][\w.]*)\((.+)\)", code, re.S):
            signatures.append(match.groups())
    assert signatures, "API reference must retain its documented signatures"
    for name, parameters in signatures:
        _assert_inline_signature(name, parameters)


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("cryoflake.validate_sequence_mask", "mask, sequence_length"),
        ("cryoflake.read_fasta", "fasta_path, use_author_chain_ids=False"),
        ("cryoflake.read_fasta", "fasta_path, use_author_chain_ids"),
        ("cryoflake.io.mrc.load_mrc", "mrc_path, multiply_global_origin=True"),
        ("cryoflake.write_artifact_list", "artifact_list_path"),
    ],
)
def test_inline_signature_checks_detect_keyword_default_and_calling_errors(
    name, parameters
):
    with pytest.raises(AssertionError):
        _assert_inline_signature(name, parameters)


def test_documented_configuration_layout_matches_packaged_yaml_files():
    example = load_examples()["config-layout"]
    root, *entries = example.code.splitlines()
    assert root == "src/cryoflake/configs/"
    directories = [(-1, Path(root))]
    files = set()
    for entry in entries:
        name = entry.strip()
        indentation = len(entry) - len(entry.lstrip())
        while directories[-1][0] >= indentation:
            directories.pop()
        path = directories[-1][1] / name
        if name.endswith("/"):
            assert (PROJECT_ROOT / path).is_dir(), example.location
            directories.append((indentation, path))
        else:
            assert (PROJECT_ROOT / path).is_file(), example.location
            files.add(path)
    assert files == {
        path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / root).rglob("*.yaml")
    }


def test_setup_commands_are_accounted_for_without_recursive_pytest_or_uv_sync():
    # Environment setup and lint/build commands run in CI itself; running them
    # inside an example would mutate the test environment or recurse into pytest.
    ci = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text()
    for example in load_examples().values():
        if example.mode != "setup":
            continue
        for command in example.shell_commands():
            if command[:3] == ["uv", "run", "cryoflake"]:
                # CLI portions and the real wheel installation have their own tests.
                continue
            if command[:3] == ["uv", "pip", "install"]:
                assert example.identifier == "guide-wheel-install"
                continue
            if command[:1] == ["cryoflake"]:
                continue
            if command[:2] == ["uv", "sync"]:
                assert command == ["uv", "sync", "--locked"] or command == [
                    "uv",
                    "sync",
                    "--locked",
                    "--dev",
                ]
                assert "uv sync --locked --dev" in ci
            elif command[:3] == ["uv", "run", "pytest"]:
                if command[3:] == ["tests/documentation"]:
                    assert " ".join(command) in ci
                    continue
                assert command[3:] in (
                    [],
                    ["--cov=cryoflake", "--cov-report=term-missing"],
                )
                assert "uv run pytest --ignore=tests/documentation" in ci
                assert "--cov=cryoflake --cov-report=term-missing" in ci
            else:
                assert " ".join(command) in ci, (example.location, command)


def _anchors(source: str) -> set[str]:
    anchors = set()
    counts = {}
    marker = None
    for line in source.splitlines():
        stripped = line.strip()
        if marker is not None:
            if fence_closes(stripped, marker):
                marker = None
            continue
        if fence := FENCE.fullmatch(stripped):
            marker = fence[1]
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if match:
            slug = re.sub(r"[^\w\- ]", "", match[1].lower()).replace(" ", "-")
            count = counts.get(slug, 0)
            counts[slug] = count + 1
            anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


def test_public_documentation_links_and_heading_anchors_resolve():
    for path in documentation_paths():
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            target = target.split(' "', 1)[0].strip("<>")
            if urlsplit(target).scheme:
                continue
            name, _, fragment = unquote(target).partition("#")
            destination = (path.parent / name).resolve() if name else path
            assert destination.exists(), f"{path}: {target}"
            if fragment and destination.suffix.lower() == ".md":
                assert fragment in _anchors(destination.read_text()), (
                    f"{path}: {target}"
                )


def test_uppercase_markdown_links_still_validate_heading_anchors(tmp_path, monkeypatch):
    from . import examples

    (tmp_path / "README.md").write_text("[Guide](docs/Guide.MD#missing)\n")
    guide = tmp_path / "docs/Guide.MD"
    guide.parent.mkdir()
    guide.write_text("# Actual heading\n")
    monkeypatch.setattr(examples, "PROJECT_ROOT", tmp_path)

    with pytest.raises(AssertionError, match="Guide.MD#missing"):
        test_public_documentation_links_and_heading_anchors_resolve()


def test_new_example_ids_cannot_hide_unexercised_code(monkeypatch):
    from . import examples

    original = examples.parse_examples

    def with_uncovered_example(path):
        parsed = original(path)
        if path == PROJECT_ROOT / "README.md":
            parsed.append(Example("untested", "python", "python", "pass\n", path, 1))
        return parsed

    monkeypatch.setattr(examples, "parse_examples", with_uncovered_example)
    with pytest.raises(AssertionError):
        test_every_documented_fence_has_a_classified_regression_scenario()
