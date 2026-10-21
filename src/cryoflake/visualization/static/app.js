/* Mol* APIs are pinned to 5.13.0. Artifact values are only rendered as text. */
"use strict";

(() => {
    const byId = id => document.getElementById(id);
    let viewer;
    let updateQueue = Promise.resolve();
    let densityTimer;

    function reportError(error) {
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
        byId("sequences").replaceChildren();
        for (const chain of summary.chains) {
            const section = document.createElement("div");
            const title = document.createElement("p");
            title.className = "sequence-heading";
            title.textContent = `${chain.chain_id} · ${chain.polymer_kind.replaceAll("_", " ")} · ${chain.residue_count} residues`;
            const sequence = document.createElement("p");
            sequence.className = "sequence";
            sequence.textContent = chain.sequence;
            section.append(title, sequence);
            byId("sequences").append(section);
        }
        if (!summary.chains.length) byId("sequences").textContent = "No polymer sequence stored in this artifact.";
        for (const limitation of summary.limitations) {
            const item = document.createElement("li");
            item.textContent = limitation;
            byId("limitations").append(item);
        }
        byId("limitations-section").hidden = !summary.limitations.length;
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
        const level = Number(byId("density-level").value);
        const opacity = Number(byId("density-opacity").value);
        const representation = viewer.plugin.managers.volume.hierarchy.current.volumes[0]?.representations[0];
        if (!representation) throw new Error("The density representation is unavailable.");
        await viewer.plugin.build().to(representation.cell).update(params => ({
            ...params,
            type: { ...params.type, params: { ...params.type.params, isoValue: { kind: "relative", relativeValue: level }, alpha: opacity } }
        })).commit();
    }

    function scheduleDensityUpdate() {
        byId("density-level-value").textContent = `${Number(byId("density-level").value).toFixed(1)} σ`;
        byId("density-opacity-value").textContent = `${Math.round(Number(byId("density-opacity").value) * 100)}%`;
        clearTimeout(densityTimer);
        densityTimer = setTimeout(() => {
            updateQueue = updateQueue.then(updateDensity).catch(reportError);
        }, 120);
    }

    async function start() {
        byId("reload").addEventListener("click", () => window.location.reload());
        const summaryReady = fetch("/summary.json").then(response => {
            if (!response.ok) throw new Error("Artifact information could not be read from the local server.");
            return response.json();
        }).then(showSummary);
        await Promise.all([summaryReady, loadEngine()]);
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
        byId("loading-message").textContent = "Loading reconstructed atomic structure…";
        await viewer.loadStructureFromUrl(new URL("/structure.cif", window.location.href).href, "mmcif", false, { label: "Reconstructed artifact" });
        byId("loading-message").textContent = "Building density overlay…";
        await viewer.loadVolumeFromUrl({ url: new URL("/density.mrc", window.location.href).href, format: "ccp4", isBinary: true }, [
            { type: "relative", value: 1, color: 0x6a9bc5, alpha: 0.25 }
        ], { entryId: "Artifact density" });
        viewer.plugin.managers.camera.reset();
        byId("show-structure").addEventListener("change", event => {
            viewer.plugin.managers.structure.hierarchy.toggleVisibility(viewer.plugin.managers.structure.hierarchy.current.structures, event.target.checked ? "show" : "hide");
        });
        byId("show-density").addEventListener("change", event => {
            viewer.plugin.managers.volume.hierarchy.toggleVisibility(viewer.plugin.managers.volume.hierarchy.current.volumes, event.target.checked ? "show" : "hide");
        });
        byId("density-level").addEventListener("input", scheduleDensityUpdate);
        byId("density-opacity").addEventListener("input", scheduleDensityUpdate);
        byId("reset-camera").addEventListener("click", () => viewer.plugin.managers.camera.reset());
        byId("display-controls").disabled = false;
        byId("loading").hidden = true;
        byId("session-status").textContent = "Structure and density aligned in artifact coordinates (Å)";
        window.addEventListener("pagehide", () => viewer.dispose(), { once: true });
    }

    start().catch(reportError);
})();
