"""Execute shipped viewer controls offline with injected browser boundaries."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.documentation
STATIC_DIR = Path(__file__).resolve().parents[2] / "src/cryoflake/visualization/static"

# Read the actual HTML and execute its packaged script. The adapter mimics browser
# range clamping/stepping and records the values sent to the renderer.
HARNESS = r"""

const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const request = JSON.parse(fs.readFileSync(0, "utf8"));
const appPath = process.argv[1];
const html = fs.readFileSync(path.join(path.dirname(appPath), "index.html"), "utf8");
const elements = {};
const pendingTimers = new Map();
const windowListeners = new Map();
const pendingCommits = [];
const calls = { volumes: [], updates: [], structures: [], visibility: [], focus: [], select: [], clipboard: [], disposals: 0, reloads: 0, maxConcurrentCommits: 0 };
let nextTimer = 0;
let activeCommits = 0;

class Element {
    constructor(attributes = {}, tagName = "") {
        this.attributes = attributes;
        this.tagName = tagName;
        this.listeners = {};
        this.children = [];
        this.textContent = "";
        this.hidden = "hidden" in attributes;
        this.disabled = "disabled" in attributes;
        this.checked = "checked" in attributes;
        this.min = attributes.min || "0";
        this.max = attributes.max || "100";
        this.step = attributes.step || "1";
        this.style = {};
        this.scrollHeight = 64;
        this.readOnly = "readonly" in attributes;
        this.selectionStart = 0;
        this.selectionEnd = 0;
        this.selectionCount = 0;
        this.value = attributes.value || "";
    }
    set value(value) {
        if (this.attributes.type === "range") {
            let number = Math.min(Number(this.max), Math.max(Number(this.min), Number(value)));
            if (this.step !== "any") {
                number = Number(this.min) + Math.round((number - Number(this.min)) / Number(this.step)) * Number(this.step);
            }
            this.storedValue = String(number);
        } else this.storedValue = String(value);
    }
    get value() { return this.storedValue; }
    focus() { calls.focus.push(this.attributes.id); }
    select() {
        calls.select.push(this.attributes.id || this.attributes["aria-label"]);
        this.selectionStart = 0;
        this.selectionEnd = this.value.length;
        this.selectionCount++;
    }
    append(...children) {
        for (const child of children) {
            if (child.tagName === "#document-fragment") {
                for (const fragmentChild of child.children) this.children.push(fragmentChild);
                child.children = [];
            } else this.children.push(child);
        }
    }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
    removeAttribute(name) { delete this.attributes[name]; }
}
for (const match of html.matchAll(/<[a-z][a-z0-9]*\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const attributes = {};
    for (const attribute of match[1].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
        attributes[attribute[1]] = attribute[2] ?? "";
    }
    elements[match[2]] = new Element(attributes, match[0].match(/^<([a-z][a-z0-9]*)/)[1]);
    elements[match[2]].textContent = html.slice(match.index + match[0].length).split("<")[0].trim();
}
const volumeHierarchy = {
    current: { volumes: [{ representations: [{ cell: "density-cell" }] }] },
    toggleVisibility: (_, visibility) => calls.visibility.push(visibility)
};
const viewer = {
    loadStructureFromUrl: async (...args) => calls.structures.push(args),
    loadVolumeFromUrl: async (...args) => calls.volumes.push(args),
    dispose: () => { calls.disposals++; },
    plugin: {
        managers: {
            volume: { hierarchy: volumeHierarchy },
            structure: { hierarchy: { current: { structures: [] }, toggleVisibility: () => {} } },
            camera: { reset: () => {} }
        },
        build: () => ({ to: cell => ({ update: updater => ({ commit: async () => {
            calls.updates.push({ cell, parameters: updater({ type: { params: { alpha: 0.25 } } }) });
            activeCommits++;
            calls.maxConcurrentCommits = Math.max(calls.maxConcurrentCommits, activeCommits);
            try {
                if (request.deferCommits) await new Promise((resolve, reject) => pendingCommits.push({ resolve, reject }));
            } finally {
                activeCommits--;
            }
        } }) }) })
    }
};
const engine = { Viewer: { create: async () => viewer } };
const context = {
    document: {
        getElementById: id => {
            if (!elements[id]) throw new Error(`Unknown HTML element ${id}`);
            return elements[id];
        },
        createElement: tagName => new Element({}, tagName),
        createDocumentFragment: () => new Element({}, "#document-fragment"),
        head: { append: script => Promise.resolve().then(() => script.onload()) }
    },
    window: {
        molstar: engine,
        location: { href: "http://127.0.0.1:8888/", reload: () => { calls.reloads++; } },
        addEventListener: (name, callback, options) => windowListeners.set(name, { callback, options })
    },
    molstar: engine,
    navigator: { clipboard: { writeText: async text => calls.clipboard.push(text) } },
    fetch: async () => ({ ok: true, json: async () => request.summary }),
    setTimeout: (callback, delay) => {
        const id = ++nextTimer;
        pendingTimers.set(id, { callback, delay });
        return id;
    },
    clearTimeout: id => pendingTimers.delete(id),
    URL
};
async function settle() {
    for (let index = 0; index < 20; index++) await new Promise(setImmediate);
}
(async () => {
    vm.runInNewContext(fs.readFileSync(appPath, "utf8"), context, { filename: appPath });
    await settle();
    const snapshots = [];
    function snapshot() {
        const state = Object.fromEntries(Object.entries(elements).map(([id, element]) => [id, {
            value: element.value, text: element.textContent, min: element.min, max: element.max,
            disabled: element.disabled, checked: element.checked, hidden: element.hidden,
            invalid: element.attributes["aria-invalid"], expanded: element.attributes["aria-expanded"],
            type: element.attributes.type
        }]));
        state.sequenceItems = elements.sequences.children.map(section => {
            const [heading, sequence] = section.children;
            return {
                heading: heading.textContent,
                tagName: sequence.tagName,
                type: sequence.type || sequence.attributes.type,
                text: sequence.textContent,
                value: sequence.value,
                readOnly: sequence.readOnly,
                selectionStart: sequence.selectionStart,
                selectionEnd: sequence.selectionEnd,
                selectionCount: sequence.selectionCount,
                wrap: sequence.wrap || sequence.attributes.wrap,
                className: sequence.className,
                title: sequence.title || sequence.attributes.title,
                label: sequence.attributes["aria-label"]
            };
        });
        state.renderer = {
            activeCommits,
            pendingCommits: pendingCommits.length,
            updateCount: calls.updates.length,
            densityTimers: [...pendingTimers.values()].filter(timer => timer.delay === 120).length
        };
        return state;
    }
    snapshots.push(snapshot());
    for (const action of request.actions || []) {
        if (action.releaseCommit) {
            const commit = pendingCommits.shift();
            if (!commit) throw new Error("No deferred renderer commit to release");
            if (action.error) commit.reject(new Error(action.error));
            else commit.resolve();
        } else if (action.windowEvent) {
            const listener = windowListeners.get(action.windowEvent);
            if (listener) {
                listener.callback({ persisted: action.persisted || false });
                if (listener.options?.once) windowListeners.delete(action.windowEvent);
            }
        } else if (action.id || "sequenceIndex" in action) {
            const element = "sequenceIndex" in action
                ? elements.sequences.children[action.sequenceIndex].children[1]
                : elements[action.id];
            if ("value" in action) element.value = action.value;
            if ("checked" in action) element.checked = action.checked;
            if ("selectionStart" in action) element.selectionStart = action.selectionStart;
            if ("selectionEnd" in action) element.selectionEnd = action.selectionEnd;
            element.listeners[action.event]({ target: element, key: action.key, preventDefault: () => {} });
        }
        if (action.flushTimers !== false) {
            for (const [id, timer] of pendingTimers) {
                if (timer.delay === 120) {
                    pendingTimers.delete(id);
                    timer.callback();
                }
            }
        }
        await settle();
        snapshots.push(snapshot());
    }
    process.stdout.write(JSON.stringify({ calls, snapshots }));
})().catch(error => { console.error(error); process.exitCode = 1; });

"""


def _run_controls(actions=(), *, chains=(), defer_commits=False):
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("Node.js 22 is required in CI for offline viewer script tests")
        pytest.skip("Node.js is needed for offline viewer script tests; CI installs it")
    result = subprocess.run(
        [node, "-e", HARNESS, str(STATIC_DIR / "app.js")],
        input=json.dumps(
            {
                "summary": {
                    "schema_version": 1,
                    "artifact_reference": "test/artifact",
                    "reconstruction_mode": "backbone",
                    "chains": chains,
                    "limitations": [],
                    "density": {"shape_zyx": [8, 9, 10], "voxel_size_angstrom": 1.5},
                },
                "actions": actions,
                "deferCommits": defer_commits,
            }
        ),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_ci_requires_node_for_offline_viewer_regressions(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda executable: None)
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match="Node.js 22 is required in CI"):
        _run_controls()


def _edit_action(event, *, value=None, key=None):
    action = {"id": "density-level-number", "event": event}
    if value is not None:
        action["value"] = value
    if key is not None:
        action["key"] = key
    return action


OPEN_EDITOR = {"id": "density-level-value", "event": "click"}


def _rendered_levels(result):
    return [
        update["parameters"]["type"]["params"]["isoValue"]
        for update in result["calls"]["updates"]
    ]


def test_sigma_readout_opens_a_focused_numeric_editor_on_click():
    result = _run_controls([OPEN_EDITOR])
    initial, editing = result["snapshots"]
    assert initial["density-level-number"]["hidden"]
    assert initial["density-level-number"]["type"] == "text"
    assert initial["density-level-unit"]["hidden"]
    assert not initial["density-level-value"]["hidden"]
    assert initial["density-level-value"]["expanded"] == "false"
    assert initial["density-level-value"]["text"] == "1.0 σ"
    assert float(initial["density-level"]["value"]) == 1
    assert result["calls"]["volumes"][0][1] == [
        {"type": "relative", "value": 1, "color": 0x6A9BC5, "alpha": 0.25}
    ]
    assert not editing["density-level-number"]["hidden"]
    assert not editing["density-level-unit"]["hidden"]
    assert editing["density-level-unit"]["text"] == "σ"
    assert editing["density-level-value"]["hidden"]
    assert editing["density-level-value"]["expanded"] == "true"
    assert result["calls"]["focus"] == ["density-level-number"]
    assert result["calls"]["select"] == ["density-level-number"]


@pytest.mark.parametrize("level", [2.75, 0, -0.35, 2.123456789012345, 14, -7])
def test_typed_sigma_values_sync_slider_and_renderer_without_quantization(level):
    result = _run_controls([OPEN_EDITOR, _edit_action("input", value=str(level))])
    editing = result["snapshots"][-1]
    assert not editing["density-level-number"]["hidden"]
    assert float(editing["density-level-number"]["value"]) == level
    assert float(editing["density-level"]["value"]) == level
    assert float(editing["density-level"]["min"]) <= level
    assert float(editing["density-level"]["max"]) >= level
    assert editing["density-level-value"]["text"].endswith(" σ")
    assert float(editing["density-level-value"]["text"].removesuffix(" σ")) == level
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": level}
    assert editing["contour-validation"]["hidden"]


def test_slider_changes_sync_the_sigma_textbox_during_an_edit():
    result = _run_controls(
        [OPEN_EDITOR, {"id": "density-level", "event": "input", "value": "3.125"}]
    )
    editing = result["snapshots"][-1]
    assert editing["density-level-number"]["value"] == "3.125"
    assert editing["density-level-value"]["text"] == "3.125 σ"
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 3.125}


@pytest.mark.parametrize("event,key", [("keydown", "Enter"), ("blur", None)])
def test_enter_or_clicking_away_closes_editor_and_keeps_chosen_level(event, key):
    result = _run_controls(
        [OPEN_EDITOR, _edit_action("input", value="2.5"), _edit_action(event, key=key)]
    )
    closed = result["snapshots"][-1]
    assert closed["density-level-number"]["hidden"]
    assert closed["density-level-unit"]["hidden"]
    assert not closed["density-level-value"]["hidden"]
    assert closed["density-level-value"]["expanded"] == "false"
    assert closed["density-level-value"]["text"] == "2.5 σ"
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 2.5}
    if key == "Enter":
        assert result["calls"]["focus"][-1] == "density-level-value"


def test_escape_restores_level_from_start_of_edit():
    result = _run_controls(
        [
            {"id": "density-level", "event": "input", "value": "2.5"},
            OPEN_EDITOR,
            _edit_action("input", value="4.75"),
            _edit_action("keydown", key="Escape"),
        ]
    )
    closed = result["snapshots"][-1]
    assert closed["density-level-number"]["hidden"]
    assert closed["density-level-number"]["value"] == "2.5"
    assert closed["density-level"]["value"] == "2.5"
    assert closed["density-level-value"]["text"] == "2.5 σ"
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 2.5}


@pytest.mark.parametrize(
    "invalid", ["", "not-a-number", "NaN", "Infinity", "1e309", "0x10"]
)
def test_invalid_text_keeps_last_valid_level_and_editor_open(invalid):
    result = _run_controls(
        [
            OPEN_EDITOR,
            _edit_action("input", value="2.5"),
            _edit_action("input", value=invalid),
            _edit_action("keydown", key="Enter"),
            _edit_action("blur"),
            {"id": "density-opacity", "event": "input", "value": "0.6"},
        ]
    )
    editing = result["snapshots"][-1]
    assert not editing["density-level-number"]["hidden"]
    assert editing["density-level-number"]["invalid"] == "true"
    assert not editing["contour-validation"]["hidden"]
    assert editing["density-level"]["value"] == "2.5"
    assert editing["density-level-value"]["text"] == "2.5 σ"
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 2.5}
    assert result["calls"]["updates"][-1]["parameters"]["type"]["params"][
        "alpha"
    ] == pytest.approx(0.6)
    assert editing["error"]["hidden"]


def test_valid_input_clears_error_and_escape_can_cancel_invalid_text():
    result = _run_controls(
        [
            OPEN_EDITOR,
            _edit_action("input", value=""),
            _edit_action("input", value="3"),
            _edit_action("input", value=""),
            _edit_action("keydown", key="Escape"),
        ]
    )
    valid = result["snapshots"][3]
    assert valid["contour-validation"]["hidden"]
    assert valid["density-level-number"]["invalid"] == "false"
    closed = result["snapshots"][-1]
    assert closed["density-level-number"]["hidden"]
    assert closed["contour-validation"]["hidden"]
    assert closed["density-level-value"]["text"] == "1 σ"
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 1}


OPEN_OPACITY_EDITOR = {"id": "density-opacity-value", "event": "click"}


def _opacity_action(event, *, value=None, key=None):
    action = {"id": "density-opacity-number", "event": event}
    if value is not None:
        action["value"] = value
    if key is not None:
        action["key"] = key
    return action


def _rendered_opacity(result):
    return result["calls"]["updates"][-1]["parameters"]["type"]["params"]["alpha"]


def test_opacity_readout_opens_a_plain_text_percentage_editor():
    result = _run_controls([OPEN_OPACITY_EDITOR])
    initial, editing = result["snapshots"]
    assert initial["density-opacity-number"]["type"] == "text"
    assert initial["density-opacity-number"]["hidden"]
    assert initial["density-opacity-unit"]["hidden"]
    assert not initial["density-opacity-value"]["hidden"]
    assert initial["density-opacity-value"]["text"] == "25%"
    assert not editing["density-opacity-number"]["hidden"]
    assert not editing["density-opacity-unit"]["hidden"]
    assert editing["density-opacity-unit"]["text"] == "%"
    assert editing["density-opacity-value"]["hidden"]
    assert editing["density-opacity-value"]["expanded"] == "true"
    assert editing["density-opacity-number"]["value"] == "25"
    assert result["calls"]["focus"] == ["density-opacity-number"]
    assert result["calls"]["select"] == ["density-opacity-number"]


@pytest.mark.parametrize("percent", [0, 100, 37.25, 0.125, 99.9999, 25.123456789])
def test_typed_opacity_percentage_syncs_fractional_slider_and_renderer(percent):
    result = _run_controls(
        [OPEN_OPACITY_EDITOR, _opacity_action("input", value=str(percent))]
    )
    editing = result["snapshots"][-1]
    assert float(editing["density-opacity-number"]["value"]) == percent
    assert float(editing["density-opacity"]["value"]) == pytest.approx(percent / 100)
    assert float(
        editing["density-opacity-value"]["text"].removesuffix("%")
    ) == pytest.approx(percent)
    assert _rendered_opacity(result) == pytest.approx(percent / 100)
    assert editing["opacity-validation"]["hidden"]
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 1}


def test_opacity_slider_synchronizes_percentage_text_without_float_noise():
    result = _run_controls(
        [
            OPEN_OPACITY_EDITOR,
            {"id": "density-opacity", "event": "input", "value": "0.29"},
        ]
    )
    editing = result["snapshots"][-1]
    assert editing["density-opacity-number"]["value"] == "29"
    assert editing["density-opacity-value"]["text"] == "29%"
    assert _rendered_opacity(result) == pytest.approx(0.29)


@pytest.mark.parametrize("event,key", [("keydown", "Enter"), ("blur", None)])
def test_opacity_edit_commits_on_enter_or_clicking_away(event, key):
    result = _run_controls(
        [
            OPEN_OPACITY_EDITOR,
            _opacity_action("input", value="37.25"),
            _opacity_action(event, key=key),
        ]
    )
    closed = result["snapshots"][-1]
    assert closed["density-opacity-number"]["hidden"]
    assert closed["density-opacity-unit"]["hidden"]
    assert not closed["density-opacity-value"]["hidden"]
    assert closed["density-opacity-value"]["expanded"] == "false"
    assert closed["density-opacity-value"]["text"] == "37.25%"
    assert _rendered_opacity(result) == pytest.approx(0.3725)


def test_escape_restores_opacity_at_start_of_edit():
    result = _run_controls(
        [
            {"id": "density-opacity", "event": "input", "value": "0.6"},
            OPEN_OPACITY_EDITOR,
            _opacity_action("input", value="37.25"),
            _opacity_action("keydown", key="Escape"),
        ]
    )
    closed = result["snapshots"][-1]
    assert closed["density-opacity-number"]["hidden"]
    assert float(closed["density-opacity-number"]["value"]) == 60
    assert float(closed["density-opacity"]["value"]) == pytest.approx(0.6)
    assert closed["density-opacity-value"]["text"] == "60%"
    assert _rendered_opacity(result) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "invalid", ["", "not-a-number", "NaN", "Infinity", "1e309", "-0.1", "100.1", "0x10"]
)
def test_invalid_opacity_preserves_last_valid_value_and_editor(invalid):
    result = _run_controls(
        [
            OPEN_OPACITY_EDITOR,
            _opacity_action("input", value="37.25"),
            _opacity_action("input", value=invalid),
            _opacity_action("keydown", key="Enter"),
            _opacity_action("blur"),
            {"id": "density-level", "event": "input", "value": "2.75"},
        ]
    )
    editing = result["snapshots"][-1]
    assert not editing["density-opacity-number"]["hidden"]
    assert editing["density-opacity-number"]["invalid"] == "true"
    assert not editing["opacity-validation"]["hidden"]
    assert editing["density-opacity-value"]["text"] == "37.25%"
    assert _rendered_opacity(result) == pytest.approx(0.3725)
    assert _rendered_levels(result)[-1] == {"kind": "relative", "relativeValue": 2.75}
    assert editing["error"]["hidden"]


def test_opacity_error_recovers_on_valid_entry_and_escape_cancels_invalid_text():
    result = _run_controls(
        [
            OPEN_OPACITY_EDITOR,
            _opacity_action("input", value=""),
            _opacity_action("input", value="37.25"),
            _opacity_action("input", value="101"),
            _opacity_action("keydown", key="Escape"),
        ]
    )
    valid = result["snapshots"][3]
    assert valid["opacity-validation"]["hidden"]
    assert valid["density-opacity-number"]["invalid"] == "false"
    closed = result["snapshots"][-1]
    assert closed["density-opacity-number"]["hidden"]
    assert closed["opacity-validation"]["hidden"]
    assert closed["density-opacity-value"]["text"] == "25%"
    assert _rendered_opacity(result) == pytest.approx(0.25)


@pytest.mark.parametrize("text,expected", [("+.5", 0.5), ("1e-3", 0.001)])
def test_sigma_textbox_accepts_signed_decimal_and_scientific_notation(text, expected):
    result = _run_controls([OPEN_EDITOR, _edit_action("input", value=text)])
    assert _rendered_levels(result)[-1] == {
        "kind": "relative",
        "relativeValue": expected,
    }
    assert result["snapshots"][-1]["contour-validation"]["hidden"]


@pytest.mark.parametrize("text,expected", [("+.5", 0.005), ("3.725e1", 0.3725)])
def test_opacity_textbox_accepts_decimal_and_scientific_percentages(text, expected):
    result = _run_controls([OPEN_OPACITY_EDITOR, _opacity_action("input", value=text)])
    assert _rendered_opacity(result) == pytest.approx(expected)
    assert result["snapshots"][-1]["opacity-validation"]["hidden"]


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize(
    "control,value,expected_text",
    [
        ("density-level", "3.14159265", "3.142"),
        ("density-level", "-0.123456", "-0.123"),
        ("density-level", "-0.00001", "0.000"),
        ("density-level", "1", "1.000"),
        ("density-level", "-3", "-3.000"),
        ("density-level", "8", "8.000"),
        ("density-opacity", "0.371234567", "37"),
        ("density-opacity", "0.374999", "37"),
        ("density-opacity", "0.375", "38"),
        ("density-opacity", "0.29", "29"),
        ("density-opacity", "0.9999999", "100"),
        ("density-opacity", "0.000001", "0"),
        ("density-opacity", "0", "0"),
        ("density-opacity", "1", "100"),
    ],
)
def test_slider_rounding_uses_each_controls_displayed_precision(
    control, value, expected_text, editing
):
    actions = [{"id": f"{control}-value", "event": "click"}] if editing else []
    actions.append({"id": control, "event": "input", "value": value})
    result = _run_controls(actions)
    chosen = result["snapshots"][-1]
    expected = float(expected_text)
    is_contour = control == "density-level"
    unit = " σ" if is_contour else "%"
    assert chosen[f"{control}-value"]["text"] == expected_text + unit
    assert chosen[f"{control}-number"]["value"] == expected_text
    assert chosen[f"{control}-number"]["hidden"] is not editing
    assert float(chosen[control]["value"]) == pytest.approx(
        expected if is_contour else expected / 100
    )
    assert _rendered_levels(result)[-1] == {
        "kind": "relative",
        "relativeValue": expected if is_contour else 1,
    }
    assert _rendered_opacity(result) == pytest.approx(
        0.25 if is_contour else expected / 100
    )


@pytest.mark.parametrize(
    "control,typed_value",
    [
        ("density-level", "2.123456789"),
        ("density-opacity", "37.123456789"),
    ],
)
def test_text_entry_after_dragging_preserves_manual_precision(control, typed_value):
    result = _run_controls(
        [
            {"id": control, "event": "input", "value": "0.123456789"},
            {"id": f"{control}-value", "event": "click"},
            {"id": f"{control}-number", "event": "input", "value": typed_value},
            {"id": f"{control}-number", "event": "keydown", "key": "Enter"},
        ]
    )
    chosen = result["snapshots"][-1]
    is_contour = control == "density-level"
    unit = " σ" if is_contour else "%"
    assert chosen[f"{control}-value"]["text"] == typed_value + unit
    assert chosen[f"{control}-number"]["hidden"]
    if is_contour:
        assert _rendered_levels(result)[-1] == {
            "kind": "relative",
            "relativeValue": float(typed_value),
        }
    else:
        assert _rendered_opacity(result) == pytest.approx(float(typed_value) / 100)


def test_a_slider_burst_renders_only_the_latest_state_after_debounce():
    actions = [
        {
            "id": "density-level",
            "event": "input",
            "value": str(index / 1000),
            "flushTimers": False,
        }
        for index in range(100)
    ]
    actions.extend(
        [
            {
                "id": "density-opacity",
                "event": "input",
                "value": "0.48",
                "flushTimers": False,
            },
            {"flushTimers": True},
        ]
    )
    result = _run_controls(actions)
    assert result["snapshots"][-2]["renderer"] == {
        "activeCommits": 0,
        "pendingCommits": 0,
        "updateCount": 0,
        "densityTimers": 1,
    }
    assert len(result["calls"]["updates"]) == 1
    assert _rendered_levels(result) == [{"kind": "relative", "relativeValue": 0.099}]
    assert _rendered_opacity(result) == pytest.approx(0.48)


@pytest.mark.parametrize("fail_first_commit", [False, True])
def test_slow_renderer_coalesces_pending_changes_and_catches_up(fail_first_commit):
    actions = [{"id": "density-level", "event": "input", "value": "1.25"}]
    actions.extend(
        {"id": "density-level", "event": "input", "value": str(index / 1000)}
        for index in range(100)
    )
    actions.extend(
        [
            {"id": "density-opacity", "event": "input", "value": "0.48"},
            {
                "releaseCommit": True,
                **({"error": "Renderer failure"} if fail_first_commit else {}),
            },
            {"releaseCommit": True},
        ]
    )
    result = _run_controls(actions, defer_commits=True)
    assert result["snapshots"][-3]["renderer"] == {
        "activeCommits": 1,
        "pendingCommits": 1,
        "updateCount": 1,
        "densityTimers": 0,
    }
    assert result["calls"]["maxConcurrentCommits"] == 1
    assert len(result["calls"]["updates"]) == 2
    assert _rendered_levels(result) == [
        {"kind": "relative", "relativeValue": 1.25},
        {"kind": "relative", "relativeValue": 0.099},
    ]
    assert _rendered_opacity(result) == pytest.approx(0.48)
    assert result["snapshots"][-1]["renderer"] == {
        "activeCommits": 0,
        "pendingCommits": 0,
        "updateCount": 2,
        "densityTimers": 0,
    }
    assert result["snapshots"][-1]["error"]["hidden"] is not fail_first_commit


def test_pagehide_cancels_a_debounced_update_before_disposal():
    result = _run_controls(
        [
            {
                "id": "density-level",
                "event": "input",
                "value": "2.5",
                "flushTimers": False,
            },
            {"windowEvent": "pagehide"},
            {"flushTimers": True},
        ]
    )
    assert result["calls"]["disposals"] == 1
    assert result["calls"]["updates"] == []
    assert result["snapshots"][-1]["renderer"]["densityTimers"] == 0
    assert result["snapshots"][-1]["error"]["hidden"]


@pytest.mark.parametrize("persisted", [False, True])
def test_browser_back_reloads_a_disposed_viewer_only_when_restored_from_cache(
    persisted,
):
    result = _run_controls(
        [
            {"windowEvent": "pageshow", "persisted": False},
            {"windowEvent": "pagehide"},
            {"windowEvent": "pageshow", "persisted": persisted},
        ]
    )
    assert result["calls"]["disposals"] == 1
    assert result["calls"]["reloads"] == int(persisted)
    assert result["calls"]["updates"] == []
    assert result["snapshots"][-1]["error"]["hidden"]


@pytest.mark.parametrize("reject_inflight", [False, True])
def test_pagehide_discards_pending_changes_and_ignores_late_commits(reject_inflight):
    result = _run_controls(
        [
            {"id": "density-level", "event": "input", "value": "2.5"},
            {"id": "density-opacity", "event": "input", "value": "0.48"},
            {
                "id": "density-level",
                "event": "input",
                "value": "3.5",
                "flushTimers": False,
            },
            {"windowEvent": "pagehide"},
            {
                "releaseCommit": True,
                **({"error": "Disposed renderer"} if reject_inflight else {}),
            },
            {"id": "density-level", "event": "input", "value": "4.5"},
            {"windowEvent": "pagehide"},
        ],
        defer_commits=True,
    )
    assert result["calls"]["disposals"] == 1
    assert len(result["calls"]["updates"]) == 1
    assert result["calls"]["maxConcurrentCommits"] == 1
    assert result["snapshots"][-1]["renderer"] == {
        "activeCommits": 0,
        "pendingCommits": 0,
        "updateCount": 1,
        "densityTimers": 0,
    }
    assert result["snapshots"][-1]["error"]["hidden"]


def _chain(sequence, *, chain_id="A", polymer_kind="protein"):
    return {
        "chain_id": chain_id,
        "polymer_kind": polymer_kind,
        "residue_count": len(sequence),
        "sequence": sequence,
    }


@pytest.mark.parametrize(
    "sequence,grouped",
    [
        ("A", "A"),
        ("ACDEF", "ACDEF"),
        ("ACDEFG", "ACDEF G"),
        ("ACDEFGHIKL", "ACDEF GHIKL"),
        ("ACDEFGHIKLM", "ACDEF GHIKL M"),
    ],
)
def test_sequences_are_grouped_every_five_residues_without_trailing_spaces(
    sequence, grouped
):
    result = _run_controls(chains=[_chain(sequence)])
    panel = result["snapshots"][0]
    assert panel["sequence-count"]["text"] == "(1)"
    assert len(panel["sequenceItems"]) == 1
    sequence_box = panel["sequenceItems"][0]
    assert sequence_box["heading"] == f"A · protein · {len(sequence)} residues"
    assert sequence_box["tagName"] == "textarea"
    assert sequence_box["className"] == "sequence"
    assert sequence_box["value"] == grouped
    assert sequence_box["readOnly"]
    assert sequence_box["wrap"] == "soft"
    assert sequence_box["title"] == "Click to select the unspaced sequence"
    assert sequence_box["label"] == "Sequence for chain A"
    assert result["calls"]["clipboard"] == []


@pytest.mark.parametrize(
    "sequence,polymer_kind,grouped",
    [
        ("ACDEXBZJUO?", "protein", "ACDEX BZJUO ?"),
        ("ACGTNACGTRY", "dna", "ACGTN ACGTR Y"),
        ("ACGUNACGURY", "rna", "ACGUN ACGUR Y"),
    ],
)
def test_click_selects_exact_sequence_without_display_grouping(
    sequence, polymer_kind, grouped
):
    result = _run_controls(
        [
            {"sequenceIndex": 0, "event": "focus"},
            {
                "sequenceIndex": 0,
                "event": "click",
                "selectionStart": 3,
                "selectionEnd": 3,
            },
        ],
        chains=[_chain(sequence, chain_id="B2", polymer_kind=polymer_kind)],
    )
    initial, focused, clicked = result["snapshots"]
    assert initial["sequenceItems"][0]["value"] == grouped
    for panel in [focused, clicked]:
        selected = panel["sequenceItems"][0]
        assert selected["value"] == sequence
        assert selected["selectionStart"] == 0
        assert selected["selectionEnd"] == len(sequence)
        assert selected["readOnly"]
        assert panel["error"]["hidden"]
    assert focused["sequenceItems"][0]["selectionCount"] == 1
    assert clicked["sequenceItems"][0]["selectionCount"] == 2
    assert result["calls"]["clipboard"] == []


def test_clicking_an_already_focused_sequence_selects_it_all_again():
    result = _run_controls(
        [
            {"sequenceIndex": 0, "event": "focus"},
            {"sequenceIndex": 0, "event": "click"},
            {
                "sequenceIndex": 0,
                "event": "click",
                "selectionStart": 2,
                "selectionEnd": 5,
            },
        ],
        chains=[_chain("ACDEFGHIKLM")],
    )
    selected = result["snapshots"][-1]["sequenceItems"][0]
    assert selected["value"] == "ACDEFGHIKLM"
    assert selected["selectionStart"] == 0
    assert selected["selectionEnd"] == 11
    assert selected["selectionCount"] == 3
    assert result["calls"]["clipboard"] == []


def test_switching_chains_regroups_previous_sequence_and_selects_the_next():
    chains = [_chain("ACDEFG", chain_id="A"), _chain("NACGUN", chain_id="B")]
    result = _run_controls(
        [
            {"sequenceIndex": 1, "event": "focus"},
            {"sequenceIndex": 1, "event": "click"},
            {"sequenceIndex": 1, "event": "blur"},
            {"sequenceIndex": 0, "event": "focus"},
            {"sequenceIndex": 0, "event": "click"},
            {"sequenceIndex": 0, "event": "blur"},
        ],
        chains=chains,
    )
    initial = result["snapshots"][0]
    assert [item["value"] for item in initial["sequenceItems"]] == [
        "ACDEF G",
        "NACGU N",
    ]
    selected_b = result["snapshots"][2]["sequenceItems"]
    assert [item["value"] for item in selected_b] == ["ACDEF G", "NACGUN"]
    assert selected_b[1]["selectionEnd"] == 6
    selected_a = result["snapshots"][5]["sequenceItems"]
    assert [item["value"] for item in selected_a] == ["ACDEFG", "NACGU N"]
    assert selected_a[0]["selectionEnd"] == 6
    regrouped = result["snapshots"][-1]["sequenceItems"]
    assert [item["value"] for item in regrouped] == ["ACDEF G", "NACGU N"]
    assert all(item["readOnly"] for item in regrouped)
    assert result["calls"]["clipboard"] == []


def test_clicking_away_regroups_sequence_and_density_controls_remain_usable():
    result = _run_controls(
        [
            {"sequenceIndex": 0, "event": "focus"},
            {"sequenceIndex": 0, "event": "click"},
            {"sequenceIndex": 0, "event": "blur"},
            {"id": "density-level", "event": "input", "value": "2.125"},
        ],
        chains=[_chain("ACDEFGHIKLM")],
    )
    assert result["calls"]["clipboard"] == []
    chosen = result["snapshots"][-1]
    assert chosen["sequenceItems"][0]["value"] == "ACDEF GHIKL M"
    assert chosen["sequenceItems"][0]["readOnly"]
    assert chosen["error"]["hidden"]
    assert _rendered_levels(result)[-1] == {
        "kind": "relative",
        "relativeValue": 2.125,
    }


def test_missing_sequences_keep_the_existing_empty_state():
    result = _run_controls()
    initial = result["snapshots"][0]
    assert initial["sequenceItems"] == []
    assert initial["sequence-count"]["text"] == "(0)"
    assert (
        initial["sequences"]["text"] == "No polymer sequence stored in this artifact."
    )
    assert result["calls"]["clipboard"] == []
