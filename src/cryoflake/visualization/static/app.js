/* Mol* APIs are pinned to 5.13.0. Artifact values are only rendered as text. */
"use strict";

(() => {
    const byId = id => document.getElementById(id);
    let viewer;
    let densityTimer;
    let densityUpdateInFlight = false;
    let densityUpdatePending = false;
    let disposed = false;
    let densityLevel = 1;
    let densityLevelAtEditStart = 1;
    let editingDensityLevel = false;
    let densityOpacityPercent = 25;
    let densityOpacityPercentAtEditStart = 25;
    let editingDensityOpacity = false;

    function reportError(error) {
        if (disposed) return;
        byId("loading").hidden = true;
        byId("error-message").textContent = error.message || String(error);
        byId("error").hidden = false;
        byId("session-status").textContent = "Visualization unavailable · downloads ready";
    }

    function showSummary(summary) {
        byId("artifact-reference").textContent = summary.artifact_reference;
        byId("artifact-title").textContent = summary.reconstruction_mode === "full_atom"
            ? "Atomic structure & density" : "Backbone & density";
        for (const [key, label] of [["atom_count", "Atoms"], ["residue_count", "Residues"], ["chain_count", "Chains"], ["component_count", "Components"]]) {
            const item = document.createElement("div");
            item.className = "stat";
            const title = document.createElement("dt");
            title.textContent = label;
            const value = document.createElement("dd");
            value.textContent = Number(summary[key] || 0).toLocaleString();
            item.append(title, value);
            byId("artifact-stats").append(item);
        }
        const density = summary.density;
        byId("geometry").textContent = `${density.shape_zyx.join(" × ")} voxels (Z, Y, X) · ${density.voxel_size_angstrom} Å / voxel`;
        byId("sequence-count").textContent = `(${summary.chains.length})`;
        const sequenceSections = document.createDocumentFragment();
        const sequenceInputs = [];
        for (const chain of summary.chains) {
            const section = document.createElement("div");
            const title = document.createElement("p");
            title.className = "sequence-heading";
            title.textContent = `${chain.chain_id} · ${chain.polymer_kind.replaceAll("_", " ")} · ${chain.residue_count} residues`;
            const sequence = document.createElement("textarea");
            sequence.readOnly = true;
            sequence.rows = 1;
            sequence.wrap = "soft";
            sequence.className = "sequence";
            sequence.title = "Click to select the unspaced sequence";
            sequence.setAttribute("aria-label", `Sequence for chain ${chain.chain_id}`);
            const groupedSequence = (chain.sequence.match(/.{1,5}/g) || []).join(" ");
            sequence.value = groupedSequence;
            const selectSequence = () => {
                sequence.value = chain.sequence;
                resizeSequence(sequence);
                sequence.select();
                sequence.scrollTop = 0;
            };
            sequence.addEventListener("focus", selectSequence);
            sequence.addEventListener("click", selectSequence);
            sequence.addEventListener("blur", () => {
                sequence.value = groupedSequence;
                resizeSequence(sequence);
                sequence.scrollTop = 0;
            });
            section.append(title, sequence);
            sequenceSections.append(section);
            sequenceInputs.push(sequence);
        }
        byId("sequences").replaceChildren(sequenceSections);
        resizeSequences(sequenceInputs);
        if (!summary.chains.length) byId("sequences").textContent = "No polymer sequence stored in this artifact.";
        for (const limitation of summary.limitations) {
            const item = document.createElement("li");
            item.textContent = limitation;
            byId("limitations").append(item);
        }
        byId("limitations-section").hidden = !summary.limitations.length;
    }

    function resizeSequence(sequence) {
        resizeSequences([sequence]);
    }

    function resizeSequences(sequences) {
        // Measure together so a large assembly does not force layout per chain.
        for (const sequence of sequences) sequence.style.height = "0px";
        const heights = sequences.map(sequence => Math.min(sequence.scrollHeight + 2, 160));
        sequences.forEach((sequence, index) => { sequence.style.height = `${heights[index]}px`; });
    }

    function loadEngine() {
        return new Promise((resolve, reject) => {
            const script = document.createElement("script");
            const timeout = setTimeout(() => reject(new Error("The Mol* engine timed out. Internet access to cdn.jsdelivr.net is required.")), 20000);
            script.src = "https://cdn.jsdelivr.net/npm/molstar@5.13.0/build/viewer/molstar.js";
            script.integrity = "sha384-AYj8yOj0eWt7L72Y+hfKBCur/Js/y66s/lrQ5FYTJBQfeAIox7gZE8MV+KS8/Koc";
            script.crossOrigin = "anonymous";
            script.onload = () => {
                clearTimeout(timeout);
                if (typeof window.molstar?.Viewer?.create !== "function") {
                    reject(new Error("The Mol* engine failed to initialize. Check your browser's script policy and reload, or use the ChimeraX viewer."));
                    return;
                }
                resolve();
            };
            script.onerror = () => { clearTimeout(timeout); reject(new Error("The Mol* engine could not load. Check your internet access to cdn.jsdelivr.net, or use the ChimeraX viewer.")); };
            document.head.append(script);
        });
    }

    async function updateDensity() {
        const level = densityLevel;
        const opacity = densityOpacityPercent / 100;
        const representation = viewer.plugin.managers.volume.hierarchy.current.volumes[0]?.representations[0];
        if (!representation) throw new Error("The density representation is unavailable.");
        await viewer.plugin.build().to(representation.cell).update(params => ({
            ...params,
            type: { ...params.type, params: { ...params.type.params, isoValue: { kind: "relative", relativeValue: level }, alpha: opacity } }
        })).commit();
    }

    function readFiniteDecimal(input) {
        const text = input.value.trim();
        const value = Number(text);
        return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text) && Number.isFinite(value) ? value : NaN;
    }

    function setDensityLevel(level, updateNumberInput, levelText = String(level)) {
        const slider = byId("density-level");
        // Keep typed values usable without clamping or rounding to slider steps.
        if (level < Number(slider.min)) slider.min = String(level);
        if (level > Number(slider.max)) slider.max = String(level);
        slider.value = String(level);
        densityLevel = level;
        if (updateNumberInput) byId("density-level-number").value = levelText;
        byId("density-level-value").textContent = `${levelText} σ`;
        byId("density-level-number").setAttribute("aria-invalid", "false");
        byId("contour-validation").hidden = true;
        scheduleDensityUpdate();
    }

    function readDensityLevelNumber() {
        const input = byId("density-level-number");
        const level = readFiniteDecimal(input);
        if (!Number.isFinite(level)) {
            input.setAttribute("aria-invalid", "true");
            byId("contour-validation").hidden = false;
            return false;
        }
        setDensityLevel(level, false);
        return true;
    }

    function beginDensityLevelEdit() {
        if (editingDensityLevel) return;
        editingDensityLevel = true;
        densityLevelAtEditStart = densityLevel;
        const input = byId("density-level-number");
        input.value = String(densityLevel);
        input.hidden = false;
        byId("density-level-unit").hidden = false;
        byId("density-level-value").hidden = true;
        byId("density-level-value").setAttribute("aria-expanded", "true");
        input.focus();
        input.select();
    }

    function closeDensityLevelEdit(restoreFocus) {
        editingDensityLevel = false;
        byId("density-level-number").hidden = true;
        byId("density-level-unit").hidden = true;
        byId("density-level-value").hidden = false;
        byId("density-level-value").setAttribute("aria-expanded", "false");
        if (restoreFocus) byId("density-level-value").focus();
    }

    function finishDensityLevelEdit(restoreFocus) {
        if (editingDensityLevel && readDensityLevelNumber()) closeDensityLevelEdit(restoreFocus);
    }

    function handleDensityLevelKey(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            finishDensityLevelEdit(true);
        } else if (event.key === "Escape") {
            event.preventDefault();
            setDensityLevel(densityLevelAtEditStart, true);
            closeDensityLevelEdit(true);
        }
    }

    function setDensityOpacity(percent, updateNumberInput) {
        densityOpacityPercent = percent;
        byId("density-opacity").value = String(percent / 100);
        if (updateNumberInput) byId("density-opacity-number").value = String(percent);
        byId("density-opacity-value").textContent = `${percent}%`;
        byId("density-opacity-number").setAttribute("aria-invalid", "false");
        byId("opacity-validation").hidden = true;
        scheduleDensityUpdate();
    }

    function readDensityOpacityNumber() {
        const input = byId("density-opacity-number");
        const percent = readFiniteDecimal(input);
        if (!Number.isFinite(percent) || percent < 0 || percent > 100) {
            input.setAttribute("aria-invalid", "true");
            byId("opacity-validation").hidden = false;
            return false;
        }
        setDensityOpacity(percent, false);
        return true;
    }

    function beginDensityOpacityEdit() {
        if (editingDensityOpacity) return;
        editingDensityOpacity = true;
        densityOpacityPercentAtEditStart = densityOpacityPercent;
        const input = byId("density-opacity-number");
        input.value = String(densityOpacityPercent);
        input.hidden = false;
        byId("density-opacity-unit").hidden = false;
        byId("density-opacity-value").hidden = true;
        byId("density-opacity-value").setAttribute("aria-expanded", "true");
        input.focus();
        input.select();
    }

    function closeDensityOpacityEdit(restoreFocus) {
        editingDensityOpacity = false;
        byId("density-opacity-number").hidden = true;
        byId("density-opacity-unit").hidden = true;
        byId("density-opacity-value").hidden = false;
        byId("density-opacity-value").setAttribute("aria-expanded", "false");
        if (restoreFocus) byId("density-opacity-value").focus();
    }

    function finishDensityOpacityEdit(restoreFocus) {
        if (editingDensityOpacity && readDensityOpacityNumber()) closeDensityOpacityEdit(restoreFocus);
    }

    function handleDensityOpacityKey(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            finishDensityOpacityEdit(true);
        } else if (event.key === "Escape") {
            event.preventDefault();
            setDensityOpacity(densityOpacityPercentAtEditStart, true);
            closeDensityOpacityEdit(true);
        }
    }

    function readDensityLevelSlider(event) {
        const level = Number(Number(event.target.value).toFixed(3));
        setDensityLevel(level, true, level.toFixed(3));
    }

    function readDensityOpacitySlider(event) {
        // Shift the slider's decimal point without binary multiplication noise.
        const [coefficient, exponent = "0"] = event.target.value.split(/e/i);
        const percent = Math.round(Number(`${coefficient}e${Number(exponent) + 2}`));
        setDensityOpacity(percent, true);
    }

    function scheduleDensityUpdate() {
        if (disposed) return;
        clearTimeout(densityTimer);
        densityTimer = setTimeout(() => {
            densityTimer = undefined;
            if (disposed) return;
            densityUpdatePending = true;
            flushDensityUpdates();
        }, 120);
    }

    async function flushDensityUpdates() {
        if (disposed || densityUpdateInFlight) return;
        densityUpdateInFlight = true;
        try {
            while (!disposed && densityUpdatePending) {
                densityUpdatePending = false;
                try {
                    await updateDensity();
                } catch (error) {
                    reportError(error);
                }
            }
        } finally {
            densityUpdateInFlight = false;
        }
    }

    async function start() {
        window.addEventListener("pageshow", event => {
            if (event.persisted) window.location.reload();
        });
        window.addEventListener("pagehide", () => {
            disposed = true;
            clearTimeout(densityTimer);
            densityUpdatePending = false;
            viewer?.dispose();
        }, { once: true });
        byId("reload").addEventListener("click", () => window.location.reload());
        const summaryReady = fetch("/summary.json").then(response => {
            if (!response.ok) throw new Error("Artifact information could not be read from the local server.");
            return response.json();
        }).then(showSummary);
        await Promise.all([summaryReady, loadEngine()]);
        if (disposed) return;
        viewer = await molstar.Viewer.create("viewer", {
            extensions: [],
            layoutIsExpanded: false,
            // Mol*'s sequence region is part of its controls layout.
            layoutShowControls: true,
            collapseRightPanel: true,
            layoutShowRemoteState: false,
            layoutShowSequence: true,
            layoutShowLog: false,
            layoutShowLeftPanel: false,
            viewportShowExpand: false,
            volumeStreamingDisabled: true,
            viewportBackgroundColor: "#f9fbfc"
        });
        if (disposed) { viewer.dispose(); return; }
        byId("loading-message").textContent = "Loading reconstructed atomic structure…";
        await viewer.loadStructureFromUrl(new URL("/structure.cif", window.location.href).href, "mmcif", false, { label: "Reconstructed artifact" });
        if (disposed) return;
        byId("loading-message").textContent = "Building density overlay…";
        await viewer.loadVolumeFromUrl({ url: new URL("/density.mrc", window.location.href).href, format: "ccp4", isBinary: true }, [
            { type: "relative", value: 1, color: 0x6a9bc5, alpha: 0.25 }
        ], { entryId: "Artifact density" });
        if (disposed) return;
        viewer.plugin.managers.camera.reset();
        byId("show-structure").addEventListener("change", event => {
            viewer.plugin.managers.structure.hierarchy.toggleVisibility(viewer.plugin.managers.structure.hierarchy.current.structures, event.target.checked ? "show" : "hide");
        });
        byId("show-density").addEventListener("change", event => {
            viewer.plugin.managers.volume.hierarchy.toggleVisibility(viewer.plugin.managers.volume.hierarchy.current.volumes, event.target.checked ? "show" : "hide");
        });
        byId("density-level").addEventListener("input", readDensityLevelSlider);
        byId("density-level-value").addEventListener("click", beginDensityLevelEdit);
        byId("density-level-number").addEventListener("input", readDensityLevelNumber);
        byId("density-level-number").addEventListener("keydown", handleDensityLevelKey);
        byId("density-level-number").addEventListener("blur", () => finishDensityLevelEdit(false));
        byId("density-opacity").addEventListener("input", readDensityOpacitySlider);
        byId("density-opacity-value").addEventListener("click", beginDensityOpacityEdit);
        byId("density-opacity-number").addEventListener("input", readDensityOpacityNumber);
        byId("density-opacity-number").addEventListener("keydown", handleDensityOpacityKey);
        byId("density-opacity-number").addEventListener("blur", () => finishDensityOpacityEdit(false));
        byId("reset-camera").addEventListener("click", () => viewer.plugin.managers.camera.reset());
        byId("display-controls").disabled = false;
        byId("loading").hidden = true;
        byId("session-status").textContent = "Structure and density aligned in artifact coordinates (Å)";
    }

    start().catch(reportError);
})();
