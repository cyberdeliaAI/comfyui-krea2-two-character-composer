import { app } from "../../scripts/app.js";

const NODE_TYPE = "Krea2TwoCharacterComposer";
const FREE = "Free boxes";
const DEFAULT_BOXES = "5, 10, 42, 85; 53, 10, 42, 85";
const FIELDS = [
    ["trigger", "Trigger"],
    ["prompt", "Prompt"],
    ["lora", "LoRA"],
    ["strength", "LoRA strength"],
];
// Widget order of 0.1.x / 0.2.x, used to migrate saved workflows.
const LEGACY_ORDER = [
    "direction", "character_1_percent", "base_prompt",
    "character_1_prompt", "character_1_lora", "character_1_strength",
    "character_2_prompt", "character_2_lora", "character_2_strength",
    "separation",
];
const CHARACTER_ORDER = [1, 2].flatMap((number) =>
    ["trigger", "prompt", "lora", "strength"].map((suffix) => `character_${number}_${suffix}`));
const PREVIOUS_ORDER = [
    "direction", "character_1_percent", "base_prompt", "negative_prompt",
    ...CHARACTER_ORDER, "separation", "boxes", "lora_edge_softness", "base_strength", "preview_ratio",
];
const CURRENT_ORDER = [
    "base_prompt", "negative_prompt", ...CHARACTER_ORDER,
    "direction", "character_1_percent", "separation", "boxes", "lora_edge_softness", "base_strength", "preview_ratio",
];
const LAYOUT_PROPERTY = "krea2_composer_widget_layout";
const NEGATIVE_HEIGHT = 64;
// Fields added after 0.3 and their defaults.
const ADDED = { boxes: DEFAULT_BOXES, lora_edge_softness: 0, base_strength: 1.0, preview_ratio: "Auto" };
const COLORS = ["#224e70", "#684378"];
const HANDLE = 9;
const MIN_SIZE = 5;

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function restoreEmptyBoxes(node) {
    const boxes = widget(node, "boxes");
    // A connected text input owns the coordinates; its saved placeholder may
    // be empty and must not be replaced by local defaults.
    if (!boxes || node.inputs?.some((input) => input.name === "boxes" && input.link != null)) return;
    if (boxes.value == null || (typeof boxes.value === "string" && !boxes.value.trim())) {
        boxes.value = DEFAULT_BOXES;
    }
}

function installCompactStyles() {
    if (typeof document === "undefined" || document.getElementById("krea2-composer-compact-negative")) return;
    const style = document.createElement("style");
    style.id = "krea2-composer-compact-negative";
    // Nodes 2.0 renders a native textarea instead of the original DOM element
    // and filters style options. Scope the height to this node type and field.
    style.textContent = `
        [data-widget-name="negative_prompt"]:has([node-type="${NODE_TYPE}"]) {
            height: ${NEGATIVE_HEIGHT}px;
            min-height: ${NEGATIVE_HEIGHT}px;
            max-height: ${NEGATIVE_HEIGHT}px;
            align-self: start;
        }
        [data-widget-name="negative_prompt"] [node-type="${NODE_TYPE}"],
        [data-widget-name="negative_prompt"] [node-type="${NODE_TYPE}"] textarea {
            height: ${NEGATIVE_HEIGHT}px !important;
            min-height: ${NEGATIVE_HEIGHT}px !important;
            max-height: ${NEGATIVE_HEIGHT}px !important;
        }
    `;
    document.head.appendChild(style);
    // Vue uses auto grid tracks for multiline controls. Limit this one track
    // as well, so making the node taller cannot leave a large empty row below
    // the compact textarea. Re-apply when Vue rebuilds its inline row layout.
    const fitGrid = (grid) => {
        const rows = [...grid.querySelectorAll("[data-widget-name]")]
            .filter((row) => row.closest(".lg-node-widgets") === grid);
        const index = rows.findIndex((row) => row.dataset.widgetName === "negative_prompt"
            && row.querySelector(`[node-type="${NODE_TYPE}"]`));
        const tracks = grid.style.gridTemplateRows.trim().split(/\s+/);
        if (index < 0 || tracks.length !== rows.length) return;
        tracks[index] = `${NEGATIVE_HEIGHT}px`;
        const size = tracks.join(" ");
        if (grid.style.gridTemplateRows !== size) grid.style.gridTemplateRows = size;
    };
    document.querySelectorAll(".lg-node-widgets").forEach(fitGrid);
    if (typeof MutationObserver !== "undefined") {
        const observer = new MutationObserver((records) => {
            const grids = new Set();
            for (const record of records) {
                const parent = record.target.closest?.(".lg-node-widgets");
                if (parent) grids.add(parent);
                if (record.type !== "childList") continue;
                for (const added of record.addedNodes) {
                    if (added.matches?.(".lg-node-widgets")) grids.add(added);
                    for (const grid of added.querySelectorAll?.(".lg-node-widgets") ?? []) grids.add(grid);
                }
            }
            grids.forEach(fitGrid);
        });
        observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["style"] });
    }
}

function compactNegative(node) {
    const negative = widget(node, "negative_prompt");
    if (!negative || String(negative.type).startsWith("converted-widget")) return;
    negative.options ??= {};
    Object.assign(negative.options, {
        getMinHeight: () => NEGATIVE_HEIGHT,
        getMaxHeight: () => NEGATIVE_HEIGHT,
        getHeight: () => NEGATIVE_HEIGHT,
        rows: 3,
    });
    // Canvas nodes use computeSize/layout size. Native Vue textareas are sized
    // separately by the scoped stylesheet above.
    negative.computeSize = () => [320, NEGATIVE_HEIGHT];
    negative.computeLayoutSize = () => ({ minHeight: NEGATIVE_HEIGHT, maxHeight: NEGATIVE_HEIGHT, minWidth: 0 });
    negative.computedHeight = NEGATIVE_HEIGHT;
    const element = negative.element ?? negative.inputEl;
    if (element) {
        element.rows = 3;
        for (const name of ["--comfy-widget-min-height", "--comfy-widget-max-height", "--comfy-widget-height"]) {
            element.style?.setProperty(name, `${NEGATIVE_HEIGHT}px`);
        }
    }
}

function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
}

function values(node) {
    const raw = Number(widget(node, "character_1_percent")?.value ?? 50);
    const direction = widget(node, "direction")?.value;
    return {
        percent: Number.isFinite(raw) ? clamp(Math.round(raw), 1, 99) : 50,
        vertical: direction === "Top / Bottom",
        free: direction === FREE,
    };
}

function parseBoxes(text) {
    const parts = String(text ?? "").split(/[;\n|]/).filter((part) => part.trim());
    if (parts.length !== 2) return null;
    const boxes = parts.map((part) => (part.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number));
    if (boxes.some((b) => b.length !== 4)) return null;
    return boxes.map(([x, y, w, h]) => ({ x, y, w, h }));
}

function formatBoxes(boxes) {
    const r = (v) => Math.round(v * 10) / 10;
    return boxes.map((b) => `${r(b.x)}, ${r(b.y)}, ${r(b.w)}, ${r(b.h)}`).join("; ");
}

function maskConnected(node, number) {
    return node.inputs?.find((input) => input.name === `character_${number}_mask`)?.link != null;
}

function positiveNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : null;
}

function inputSource(node, name, graph) {
    const link = node.inputs?.find((input) => input.name === name)?.link;
    if (link == null) return null;
    const info = graph?.links?.[link] ?? graph?.links?.get?.(link);
    const source = graph?.getNodeById?.(info?.origin_id);
    return source ? { node: source, slot: info.origin_slot } : null;
}

function numericField(node, name, graph, depth = 0, dimension = name) {
    if (depth > 8) return null;
    const input = inputSource(node, name, graph);
    if (input) {
        const source = input.node;
        const sourceGraph = source.graph ?? graph;
        const outputName = source.outputs?.[input.slot]?.name?.toLowerCase();
        const field = ["width", "height"].includes(outputName) ? outputName : dimension;
        const direct = numericField(source, field, sourceGraph, depth + 1, field);
        if (direct) return direct;
        const primitive = source.widgets?.find((item) => ["value", "number"].includes(item.name?.toLowerCase()));
        if (primitive) return positiveNumber(primitive.value);
        const dimensions = resolutionLabel(source);
        if (dimensions) return field === "width" ? dimensions.width : dimensions.height;
        if (source.type === "Reroute" && source.inputs?.length === 1) return numericField(source, source.inputs[0].name, sourceGraph, depth + 1, field);
        return null;
    }
    return positiveNumber(widget(node, name)?.value);
}

function resolutionLabel(node) {
    // Resolution selectors such as CyberKrea show 1152×1440 (4:5).
    for (const name of ["resolution", "dimensions"]) {
        const text = widget(node, name)?.value;
        const match = typeof text === "string" ? text.match(/\b(\d+)\s*[x×]\s*(\d+)\b/i) : null;
        if (match && Number(match[1]) > 0 && Number(match[2]) > 0) {
            return { width: Number(match[1]), height: Number(match[2]) };
        }
    }
    return null;
}

function nodeDimensions(node, graph, depth = 0) {
    if (!node || depth > 8) return null;
    graph = node.graph ?? graph;
    const width = numericField(node, "width", graph);
    const height = numericField(node, "height", graph);
    if (width && height) return { width, height };
    const selected = resolutionLabel(node);
    if (selected) return selected;
    const latents = node.inputs?.filter((input) => input.type === "LATENT") ?? [];
    const passthrough = latents.length === 1 ? latents[0] : node.type === "Reroute" && node.inputs?.length === 1 ? node.inputs[0] : null;
    return passthrough ? nodeDimensions(inputSource(node, passthrough.name, graph)?.node, graph, depth + 1) : null;
}

function previewAspect(node) {
    const selected = widget(node, "preview_ratio")?.value ?? "Auto";
    const ratio = String(selected).match(/^(\d+):(\d+)$/);
    if (ratio) return { aspect: Number(ratio[2]) / Number(ratio[1]), label: `Preview ${selected}` };
    try {
        const graph = node.graph ?? app.graph;
        const width = numericField(node, "width", graph);
        const height = numericField(node, "height", graph);
        const dimensions = width && height ? { width, height }
            : nodeDimensions(inputSource(node, "latent", graph)?.node, graph);
        if (dimensions) return { aspect: dimensions.height / dimensions.width, label: `${dimensions.width}×${dimensions.height} · Auto` };
    } catch {
        // Arbitrary calculated dimensions cannot be known before execution.
    }
    return { aspect: 1, label: "Auto: size unavailable · choose Preview ratio" };
}

function frameSize(node, width) {
    const { aspect } = previewAspect(node);
    const availableWidth = Math.max(1, width - 28);
    // Fit both dimensions into the available space. Clamping only height
    // would turn a square or portrait latent into a landscape preview.
    const w = Math.min(availableWidth, 260 / aspect);
    return { w, h: w * aspect };
}

function updateLabels(node) {
    const { percent, vertical, free } = values(node);
    const sides = free ? ["Box 1", "Box 2"] : vertical ? ["Top", "Bottom"] : ["Left", "Right"];
    for (const [index, side] of sides.entries()) {
        const number = index + 1;
        const place = maskConnected(node, number) ? "Mask" : side;
        for (const [suffix, title] of FIELDS) {
            const item = widget(node, `character_${number}_${suffix}`);
            if (item) item.label = `${place} · Character ${number} · ${title}`;
        }
    }
    const percentWidget = widget(node, "character_1_percent");
    if (percentWidget) {
        percentWidget.label = free
            ? "Split share (not used with Free boxes)"
            : `${sides[0]} ${percent}% / ${sides[1]} ${100 - percent}%`;
    }
    const boxesWidget = widget(node, "boxes");
    if (boxesWidget) boxesWidget.label = free ? "Free boxes · x, y, w, h (%)" : "Free boxes (not used)";
    const ratioWidget = widget(node, "preview_ratio");
    if (ratioWidget) ratioWidget.label = "Preview ratio";
    const preview = widget(node, "region_preview");
    if (preview) {
        preview.computedHeight = preview.computeSize(node.size?.[0] ?? 440)[1];
        preview.triggerDraw?.(); // Nodes 2.0 owns a separate canvas for this widget.
    }
    const minimumHeight = node.computeSize?.()[1];
    if (node.size && minimumHeight > node.size[1]) node.setSize?.([node.size[0], minimumHeight]);
    node.setDirtyCanvas?.(true, true);
}

// Refresh Auto when a visible upstream dimension changes. Each wrapper keeps
// the original callback and can notify more than one composer.
const dimensionWatchers = new WeakMap();
const subscriptions = new WeakMap();

function watchDimensions(node) {
    for (const listeners of subscriptions.get(node) ?? []) listeners.delete(node);
    const subscribed = [];
    const seen = new Set();
    function visit(source, graph, depth) {
        if (!source || seen.has(source) || depth > 8) return;
        seen.add(source);
        for (const item of source.widgets ?? []) {
            if (!["width", "height", "resolution", "dimensions", "value", "number"].includes(item.name)) continue;
            let listeners = dimensionWatchers.get(item);
            if (!listeners) {
                listeners = new Set();
                dimensionWatchers.set(item, listeners);
                const original = item.callback;
                item.callback = function (...args) {
                    const result = original?.apply(this, args);
                    for (const owner of listeners) updateLabels(owner);
                    return result;
                };
            }
            listeners.add(node);
            subscribed.push(listeners);
        }
        for (const input of source.inputs ?? []) {
            if (!["LATENT", "INT"].includes(input.type) && source.type !== "Reroute") continue;
            visit(inputSource(source, input.name, source.graph ?? graph)?.node, graph, depth + 1);
        }
    }
    const graph = node.graph ?? app.graph;
    for (const name of ["latent", "width", "height"]) visit(inputSource(node, name, graph)?.node, graph, 0);
    subscriptions.set(node, subscribed);
}

function swapCharacters(node) {
    // Both characters trade places. In a split each keeps its share; with
    // Free boxes they trade boxes.
    for (const [suffix] of FIELDS) {
        const first = widget(node, `character_1_${suffix}`);
        const second = widget(node, `character_2_${suffix}`);
        if (!first || !second) continue;
        [first.value, second.value] = [second.value, first.value];
    }
    if (!values(node).free) {
        const percentWidget = widget(node, "character_1_percent");
        if (percentWidget) percentWidget.value = 100 - values(node).percent;
    }
    updateLabels(node);
}

function migrateSavedValues(node, saved, layout, named) {
    // Modern ComfyUI saves values by name as well as position. Names survive
    // layout changes and take priority over a stale positional array.
    if (named && typeof named === "object" && !Array.isArray(named)
        && CURRENT_ORDER.some((name) => Object.hasOwn(named, name))) {
        for (const name of CURRENT_ORDER) {
            const item = widget(node, name);
            if (item && Object.hasOwn(named, name)) item.value = named[name];
        }
        for (const name of ["negative_prompt", "character_1_trigger", "character_2_trigger"]) {
            const item = widget(node, name);
            if (item && !Object.hasOwn(named, name)) item.value = "";
        }
        for (const [name, fallback] of Object.entries(ADDED)) {
            const item = widget(node, name);
            if (item && (!Object.hasOwn(named, name) || typeof item.value !== typeof fallback)) item.value = fallback;
        }
        return "named";
    }
    if (!Array.isArray(saved)) return "none";
    // 0.4.3 moves Direction/share after the characters. The layout marker
    // survives save/load and also disambiguates connected controls saved as
    // null. Unmarked pre-0.4.3 workflows use their original positional order.
    const current = layout === 2 || (saved.length >= 13 && typeof saved[1] === "string");
    const legacy = !current && saved.length >= LEGACY_ORDER.length && saved.length < 13;
    const order = current ? CURRENT_ORDER : legacy ? LEGACY_ORDER : PREVIOUS_ORDER;
    const kind = current ? "current" : legacy ? "0.2" : "0.3/0.4";
    for (const [index, name] of order.entries()) {
        const item = widget(node, name);
        if (item && index < saved.length) item.value = saved[index];
    }
    if (legacy) {
        for (const name of ["negative_prompt", "character_1_trigger", "character_2_trigger"]) {
            const item = widget(node, name);
            if (item) item.value = "";
        }
    }
    // Fields that didn't exist yet, or got a stray value, fall back to defaults.
    for (const [name, fallback] of Object.entries(ADDED)) {
        const item = widget(node, name);
        if (item && typeof item.value !== typeof fallback) item.value = fallback;
    }
    return kind;
}

function drawSplit(ctx, node, f) {
    const { percent, vertical } = values(node);
    const { x, top, w, h } = f;
    const fraction = percent / 100;
    ctx.fillStyle = COLORS[0];
    ctx.fillRect(x, top, vertical ? w : w * fraction, vertical ? h * fraction : h);
    ctx.fillStyle = COLORS[1];
    ctx.fillRect(vertical ? x : x + w * fraction, vertical ? top + h * fraction : top,
        vertical ? w : w * (1 - fraction), vertical ? h * (1 - fraction) : h);
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, top, w, h);
    ctx.beginPath();
    ctx.moveTo(vertical ? x : x + w * fraction, vertical ? top + h * fraction : top);
    ctx.lineTo(vertical ? x + w : x + w * fraction, vertical ? top + h * fraction : top + h);
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    const centers = vertical
        ? [[x + w / 2, top + h * fraction / 2], [x + w / 2, top + h * (1 + fraction) / 2]]
        : [[x + w * fraction / 2, top + h / 2], [x + w * (1 + fraction) / 2, top + h / 2]];
    // Small regions keep their colour; full labels go in the footer so text
    // never spills across a border.
    for (let i = 0; i < 2; i++) {
        const size = vertical ? h * (i ? 1 - fraction : fraction) : w * (i ? 1 - fraction : fraction);
        if (size >= 20) ctx.fillText(String(i + 1), centers[i][0], centers[i][1] + 4);
    }
    return `${vertical ? "Top" : "Left"}: 1 (${percent}%)   ·   ${vertical ? "Bottom" : "Right"}: 2 (${100 - percent}%)`;
}

function boxRect(f, b) {
    return [f.x + f.w * b.x / 100, f.top + f.h * b.y / 100, f.w * b.w / 100, f.h * b.h / 100];
}

function drawBoxes(ctx, node, f) {
    const boxes = parseBoxes(widget(node, "boxes")?.value);
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(f.x, f.top, f.w, f.h);
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.strokeRect(f.x, f.top, f.w, f.h);
    if (!boxes) {
        ctx.fillStyle = "#fca5a5";
        ctx.textAlign = "center";
        ctx.fillText("Boxes field: use 'x, y, w, h; x, y, w, h'", f.x + f.w / 2, f.top + f.h / 2);
        return "Fix the boxes field to see the layout";
    }
    // Character 2 first so Character 1 is drawn in front, as in the engine.
    for (const i of [1, 0]) {
        const [bx, by, bw, bh] = boxRect(f, boxes[i]);
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = COLORS[i];
        ctx.fillRect(bx, by, bw, bh);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#e2e8f0";
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = "#e2e8f0";
        for (const [hx, hy] of [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]]) {
            ctx.fillRect(hx - 3, hy - 3, 6, 6);
        }
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        if (bw >= 16 && bh >= 16) ctx.fillText(String(i + 1), bx + bw / 2, by + bh / 2 + 4);
    }
    return "Drag a box to move it, a corner to resize";
}

function drawPreview(ctx, node, width, y, preview) {
    const size = frameSize(node, width);
    const f = { x: (width - size.w) / 2, top: y + 24, ...size };
    preview.frame = f;
    ctx.save();
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.textAlign = "left";
    ctx.fillText(`Region layout · ${previewAspect(node).label}`, 14, y + 15);
    let footer = values(node).free ? drawBoxes(ctx, node, f) : drawSplit(ctx, node, f);
    const masked = [1, 2].filter((n) => maskConnected(node, n));
    if (masked.length) footer = `Mask input: ${masked.map((n) => `Character ${n}`).join(" and ")}`;
    ctx.fillStyle = "#cbd5e1";
    ctx.textAlign = "center";
    ctx.fillText(footer, width / 2, f.top + f.h + 18);
    ctx.restore();
}

function pointerPhase(event) {
    const type = String(event?.type ?? "");
    if (type.endsWith("down")) return "down";
    if (type.endsWith("move")) return "move";
    if (type.endsWith("up")) return "up";
    return "";
}

function previewMouse(preview, event, pos, node) {
    // Drag boxes in Free boxes mode: inside a box moves it, a corner resizes.
    const phase = pointerPhase(event);
    const f = preview.frame;
    const boxesWidget = widget(node, "boxes");
    if (!f || !boxesWidget || !values(node).free) return false;
    const u = (pos[0] - f.x) / f.w * 100;
    const v = (pos[1] - f.top) / f.h * 100;
    if (phase === "down") {
        preview.drag = null;
        const boxes = parseBoxes(boxesWidget.value);
        if (!boxes) return false;
        for (const index of [0, 1]) {  // Character 1 is in front
            const [bx, by, bw, bh] = boxRect(f, boxes[index]);
            const corners = { nw: [bx, by], ne: [bx + bw, by], sw: [bx, by + bh], se: [bx + bw, by + bh] };
            for (const [corner, [cx, cy]] of Object.entries(corners)) {
                if (Math.abs(pos[0] - cx) <= HANDLE && Math.abs(pos[1] - cy) <= HANDLE) {
                    preview.drag = { index, mode: corner, start: [u, v], box: { ...boxes[index] }, boxes };
                    return true;
                }
            }
            if (pos[0] >= bx && pos[0] <= bx + bw && pos[1] >= by && pos[1] <= by + bh) {
                preview.drag = { index, mode: "move", start: [u, v], box: { ...boxes[index] }, boxes };
                return true;
            }
        }
        return false;
    }
    const drag = preview.drag;
    if (!drag) return false;
    if (phase === "up") {
        // The release keeps the last moved position. Nodes 2.0 reports the
        // release point relative to the node instead of the widget.
        preview.drag = null;
        boxesWidget.callback?.(boxesWidget.value);
        updateLabels(node);
        return true;
    }
    if (phase === "move") {
        const du = u - drag.start[0], dv = v - drag.start[1];
        const b = { ...drag.box };
        if (drag.mode === "move") {
            b.x = clamp(b.x + du, 0, 100 - b.w);
            b.y = clamp(b.y + dv, 0, 100 - b.h);
        } else {
            let [x0, y0, x1, y1] = [b.x, b.y, b.x + b.w, b.y + b.h];
            if (drag.mode.includes("w")) x0 = clamp(x0 + du, 0, x1 - MIN_SIZE);
            if (drag.mode.includes("e")) x1 = clamp(x1 + du, x0 + MIN_SIZE, 100);
            if (drag.mode.includes("n")) y0 = clamp(y0 + dv, 0, y1 - MIN_SIZE);
            if (drag.mode.includes("s")) y1 = clamp(y1 + dv, y0 + MIN_SIZE, 100);
            Object.assign(b, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
        }
        const boxes = drag.boxes.map((item, i) => (i === drag.index ? b : item));
        boxesWidget.value = formatBoxes(boxes);
        updateLabels(node);
        return true;
    }
    return false;
}

app.registerExtension({
    name: "Krea2.SimpleTwoCharacterComposer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        installCompactStyles();
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this.properties ??= {};
            this.properties[LAYOUT_PROPERTY] = 2;
            restoreEmptyBoxes(this);
            compactNegative(this);
            // Visual-only widgets stay at the END so they can never shift the
            // positions of saved widget values.
            const swap = this.addWidget?.("button", "Swap characters", null, () => swapCharacters(this), { serialize: false });
            if (swap) swap.serialize = false;
            const node = this;
            const preview = {
                name: "region_preview",
                type: "krea2_split_preview",
                options: { serialize: false },
                serialize: false,
                frame: null,
                drag: null,
                computeSize(width) {
                    return [320, frameSize(node, width ?? node.size?.[0] ?? 440).h + 50];
                },
                draw(ctx, owner, width, y) {
                    drawPreview(ctx, owner, width, y, this);
                },
                mouse(event, pos, owner) {
                    return previewMouse(this, event, pos, owner);
                },
            };
            this.addCustomWidget(preview);
            for (const name of ["direction", "character_1_percent", "boxes", "preview_ratio"]) {
                const item = widget(this, name);
                if (!item) continue;
                const original = item.callback;
                item.callback = (...args) => {
                    const value = original?.apply(item, args);
                    if (name === "boxes" || (name === "direction" && values(this).free)) restoreEmptyBoxes(this);
                    updateLabels(this);
                    return value;
                };
            }
            this.setSize([Math.max(this.size[0], 440), Math.max(this.size[1], this.computeSize()[1])]);
            watchDimensions(this);
            updateLabels(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = configured?.apply(this, arguments);
            migrateSavedValues(this, info?.widgets_values, info?.properties?.[LAYOUT_PROPERTY], info?.widgets_values_named);
            restoreEmptyBoxes(this);
            this.properties ??= {};
            this.properties[LAYOUT_PROPERTY] = 2;
            compactNegative(this);
            watchDimensions(this);
            updateLabels(this);
            return result;
        };
        const connections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = connections?.apply(this, arguments);
            watchDimensions(this);
            updateLabels(this);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            for (const listeners of subscriptions.get(this) ?? []) listeners.delete(this);
            subscriptions.delete(this);
            return removed?.apply(this, arguments);
        };
    },
});
