import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/composer.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/, "");

const NAMES = [
    "base_prompt", "negative_prompt",
    ...[1, 2].flatMap((i) => ["trigger", "prompt", "lora", "strength"].map((s) => `character_${i}_${s}`)),
    "direction", "character_1_percent", "separation", "boxes", "lora_edge_softness", "base_strength", "preview_ratio",
];

const PREVIOUS_NAMES = ["direction", "character_1_percent", ...NAMES.slice(0, 10), ...NAMES.slice(12)];

function fixture() {
    let extension;
    vm.runInNewContext(source, { app: { registerExtension: (item) => { extension = item; } } });
    class Node {
        constructor() {
            const defaults = { direction: "Left / Right", character_1_percent: 60, separation: "Balanced", boxes: "5, 10, 42, 85; 53, 10, 42, 85", lora_edge_softness: 0, base_strength: 1, preview_ratio: "Auto" };
            this.widgets = NAMES.map((name) => ({ name, value: defaults[name] ?? "" }));
            this.size = [300, 500];
        }
        addWidget(type, name, value, callback, options) {
            const item = { type, name, value, callback, options };
            this.widgets.push(item);
            return item;
        }
        addCustomWidget(item) { this.widgets.push(item); }
        setDirtyCanvas() { this.dirty = true; }
        setSize(size) { this.size = size; }
        computeSize() { return [440, 760]; }
    }
    extension.beforeRegisterNodeDef(Node, { name: "Krea2TwoCharacterComposer" });
    const node = new Node();
    node.onNodeCreated();
    const get = (name) => node.widgets.find((w) => w.name === name);
    return { node, extension, Node, get };
}

test("direction callbacks and saved configuration refresh positional labels", () => {
    const { node, get } = fixture();
    assert.equal(get("character_1_percent").label, "Left 60% / Right 40%");
    assert.equal(get("character_1_trigger").label, "Left · Character 1 · Trigger");
    get("direction").value = "Top / Bottom";
    get("direction").callback();
    assert.match(get("character_1_prompt").label, /^Top/);
    assert.match(get("character_2_lora").label, /^Bottom/);
    get("character_1_percent").value = 35;
    node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value) });
    assert.equal(get("character_1_percent").label, "Top 35% / Bottom 65%");
});

test("visual widgets come after every saved field and are not serialized", () => {
    const { node } = fixture();
    assert.deepEqual(node.widgets.slice(0, NAMES.length).map((w) => w.name), NAMES);
    const [swap, preview] = node.widgets.slice(NAMES.length);
    assert.equal(swap.type, "button");
    assert.equal(swap.options.serialize, false);
    assert.equal(swap.serialize, false);
    assert.equal(preview.name, "region_preview");
    assert.equal(preview.options.serialize, false);
    assert.ok(node.size[0] >= 440);
});

test("swap exchanges both characters and mirrors the split", () => {
    const { node, get } = fixture();
    const values = { trigger: ["a1", "b2"], prompt: ["knight", "wizard"], lora: ["a.safetensors", "None"], strength: [0.7, 1.2] };
    for (const [suffix, [one, two]] of Object.entries(values)) {
        get(`character_1_${suffix}`).value = one;
        get(`character_2_${suffix}`).value = two;
    }
    node.widgets.find((w) => w.type === "button").callback();
    for (const [suffix, [one, two]] of Object.entries(values)) {
        assert.equal(get(`character_1_${suffix}`).value, two);
        assert.equal(get(`character_2_${suffix}`).value, one);
    }
    assert.equal(get("character_1_percent").value, 40);
    assert.equal(get("character_1_percent").label, "Left 40% / Right 60%");
});

test("0.1/0.2 workflows are migrated by field name", () => {
    const { node, get } = fixture();
    const legacy = ["Top / Bottom", 70, "garden", "trigger_a, a knight", "a.safetensors", 0.8,
        "trigger_b, a wizard", "None", 1.1, "Strong"];
    // the frontend first restores values by position...
    legacy.forEach((value, i) => { node.widgets[i].value = value; });
    // ...then onConfigure puts them back where they belong
    node.onConfigure({ widgets_values: [...legacy, null] });
    assert.equal(get("direction").value, "Top / Bottom");
    assert.equal(get("character_1_percent").value, 70);
    assert.equal(get("base_prompt").value, "garden");
    assert.equal(get("negative_prompt").value, "");
    assert.equal(get("character_1_trigger").value, "");
    assert.equal(get("character_1_prompt").value, "trigger_a, a knight");
    assert.equal(get("character_1_lora").value, "a.safetensors");
    assert.equal(get("character_2_strength").value, 1.1);
    assert.equal(get("separation").value, "Strong");
});

test("current workflows are left alone", () => {
    const { node, get } = fixture();
    get("negative_prompt").value = "blurry";
    node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value) });
    assert.equal(get("negative_prompt").value, "blurry");
});

test("canvas rectangles follow horizontal and vertical splits", () => {
    const { node, get } = fixture();
    const calls = [];
    const labels = [];
    const ctx = Object.fromEntries(["save", "restore", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke"].map((key) => [key, () => {}]));
    ctx.fillRect = (...args) => calls.push(args);
    ctx.fillText = (label) => labels.push(label);
    const preview = node.widgets.at(-1);
    preview.draw(ctx, node, 440, 0);
    assert.equal(calls[0][2], preview.frame.w * 0.6);
    assert.equal(calls[1][0], preview.frame.x + preview.frame.w * 0.6);
    get("direction").value = "Top / Bottom";
    calls.length = 0;
    preview.draw(ctx, node, 440, 0);
    assert.equal(calls[0][3], preview.frame.h * 0.6);
    assert.match(labels.at(-1), /Top: 1 \(60%\).*Bottom: 2 \(40%\)/);
});

test("extension leaves other nodes alone and keeps lifecycle callbacks", () => {
    const { extension } = fixture();
    class Other {}
    extension.beforeRegisterNodeDef(Other, { name: "KSampler" });
    assert.equal(Other.prototype.onNodeCreated, undefined);
    const events = [];
    class Original {
        onNodeCreated() { events.push("created"); return 42; }
        onConfigure() { events.push("configured"); return 7; }
    }
    extension.beforeRegisterNodeDef(Original, { name: "Krea2TwoCharacterComposer" });
    const node = new Original();
    node.widgets = [];
    node.size = [500, 900];
    node.addCustomWidget = (item) => node.widgets.push(item);
    node.setSize = () => {};
    node.computeSize = () => [500, 900];
    assert.equal(node.onNodeCreated(), 42);
    assert.equal(node.onConfigure({}), 7);
    assert.deepEqual(events, ["created", "configured"]);
});

function render(node) {
    const labels = [];
    const ctx = Object.fromEntries(["save", "restore", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke", "fillRect"].map((key) => [key, () => {}]));
    ctx.fillText = (label) => labels.push(label);
    const preview = node.widgets.find((w) => w.name === "region_preview");
    preview.draw(ctx, node, node.size[0], 0);
    return { preview, frame: preview.frame, labels };
}

function graphFor(node, sources, links) {
    node.graph = { links, getNodeById: (id) => sources[id] };
    node.onConnectionsChange();
}

test("all fixed ratios preserve portrait and landscape shapes before execution", () => {
    const { node, get } = fixture();
    for (const ratio of ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21", "5:4", "4:5"]) {
        get("preview_ratio").value = ratio;
        get("preview_ratio").callback();
        const { frame, labels } = render(node);
        const [w, h] = ratio.split(":").map(Number);
        assert.ok(Math.abs(frame.w / frame.h - w / h) < 1e-10, ratio);
        assert.ok(frame.w <= node.size[0] - 28 && frame.h <= 260 + 1e-9);
        assert.equal(frame.x, (node.size[0] - frame.w) / 2);
        assert.match(labels[0], new RegExp(ratio));
    }
});

test("ratio changes explicitly redraw the Nodes 2.0 widget and update its height", () => {
    const { node, get } = fixture();
    const preview = node.widgets.at(-1);
    let draws = 0;
    preview.triggerDraw = () => { draws++; };
    get("preview_ratio").value = "16:9";
    get("preview_ratio").callback();
    const landscape = preview.computedHeight;
    get("preview_ratio").value = "9:16";
    get("preview_ratio").callback();
    assert.equal(draws, 2);
    assert.ok(preview.computedHeight > landscape);
    assert.equal(preview.computedHeight, 310);
});

test("Auto reads a standard latent and changes when its visible dimensions change", () => {
    const { node } = fixture();
    const source = { widgets: [{ name: "width", value: 1024 }, { name: "height", value: 1536 }], inputs: [] };
    node.inputs = [{ name: "latent", type: "LATENT", link: 1 }];
    graphFor(node, { 4: source }, { 1: { origin_id: 4, origin_slot: 0 } });
    const { preview, frame } = render(node);
    assert.ok(Math.abs(frame.w / frame.h - 2 / 3) < 1e-10);
    let draws = 0;
    preview.triggerDraw = () => { draws++; };
    source.widgets[1].value = 512;
    source.widgets[1].callback();
    assert.equal(draws, 1);
    assert.equal(render(node).frame.w / render(node).frame.h, 2);
});

test("Auto recognizes CyberKrea resolution labels and manual ratio overrides them", () => {
    const { node, get } = fixture();
    const source = { widgets: [{ name: "resolution", value: "1152×1440 (4:5)" }], outputs: [{ name: "latent" }, { name: "width" }, { name: "height" }] };
    node.inputs = [{ name: "latent", type: "LATENT", link: 1 }];
    graphFor(node, { 4: source }, new Map([[1, { origin_id: 4, origin_slot: 0 }]]));
    assert.equal(render(node).frame.w / render(node).frame.h, 4 / 5);
    get("preview_ratio").value = "16:9";
    get("preview_ratio").callback();
    source.widgets[0].value = "864x2016 (9:21)";
    source.widgets[0].callback();
    assert.equal(render(node).frame.w / render(node).frame.h, 16 / 9);
    assert.equal(source.widgets[0].value, "864x2016 (9:21)");
});

test("optional INT dimensions retain width and height through reroutes", () => {
    const { node } = fixture();
    const source = { widgets: [{ name: "resolution", value: "1152x1440 (4:5)" }], outputs: [{ name: "width" }, { name: "height" }] };
    const reroute = { type: "Reroute", inputs: [{ name: "", link: 3 }], outputs: [{ name: "" }] };
    node.inputs = [{ name: "width", type: "INT", link: 1 }, { name: "height", type: "INT", link: 2 }];
    graphFor(node, { 4: source, 5: reroute }, {
        1: { origin_id: 4, origin_slot: 0 }, 2: { origin_id: 5, origin_slot: 0 }, 3: { origin_id: 4, origin_slot: 1 },
    });
    assert.equal(render(node).frame.w / render(node).frame.h, 4 / 5);
});

test("Auto follows linked primitive width and height on an empty latent", () => {
    const { node } = fixture();
    const latent = { widgets: [{ name: "width", value: 512 }, { name: "height", value: 512 }], inputs: [{ name: "width", link: 2 }, { name: "height", link: 3 }] };
    node.inputs = [{ name: "latent", type: "LATENT", link: 1 }];
    graphFor(node, { 4: latent, 5: { widgets: [{ name: "value", value: 1920 }] }, 6: { widgets: [{ name: "value", value: 1080 }] } }, {
        1: { origin_id: 4, origin_slot: 0 }, 2: { origin_id: 5, origin_slot: 0 }, 3: { origin_id: 6, origin_slot: 0 },
    });
    assert.equal(render(node).frame.w / render(node).frame.h, 16 / 9);
});

test("unknown calculated dimensions and cycles have an explicitly marked square fallback", () => {
    const { node } = fixture();
    const reroute = { type: "Reroute", inputs: [{ name: "latent", type: "LATENT", link: 1 }], widgets: [] };
    node.inputs = [{ name: "latent", type: "LATENT", link: 1 }];
    graphFor(node, { 4: reroute }, { 1: { origin_id: 4, origin_slot: 0 } });
    const { frame, labels } = render(node);
    assert.equal(frame.w, frame.h);
    assert.match(labels[0], /size unavailable/);
});

test("0.4 workflows reset the new ratio after a positional visual-widget null", () => {
    const { node, get } = fixture();
    get("preview_ratio").value = null;
    const saved = PREVIOUS_NAMES.slice(0, 16).map((n) => get(n).value);
    node.onConfigure({ widgets_values: [...saved, null] });
    assert.equal(get("preview_ratio").value, "Auto");
    assert.equal(get("boxes").value, saved[13]);
    assert.equal(get("base_strength").value, saved[15]);
});

test("portrait free-box drag keeps the last movement on Nodes 2.0 release", () => {
    const { node, get } = fixture();
    get("direction").value = "Free boxes";
    get("preview_ratio").value = "4:5";
    const { preview, frame } = render(node);
    const point = (x, y) => [frame.x + frame.w * x / 100, frame.top + frame.h * y / 100];
    assert.equal(preview.mouse({ type: "pointerdown" }, point(25, 40), node), true);
    assert.equal(preview.mouse({ type: "pointermove" }, point(30, 45), node), true);
    const moved = get("boxes").value;
    assert.match(moved, /^10, 15, 42, 85;/);
    preview.mouse({ type: "pointerup" }, point(30, 1000), node);
    assert.equal(get("boxes").value, moved);
    assert.equal(preview.drag, null);
});

test("disconnecting and removing a composer unsubscribe from source callbacks", () => {
    const { node } = fixture();
    let originals = 0;
    const source = { widgets: [{ name: "width", value: 1024, callback() { originals++; } }, { name: "height", value: 512 }] };
    node.inputs = [{ name: "latent", type: "LATENT", link: 1 }];
    graphFor(node, { 4: source }, { 1: { origin_id: 4, origin_slot: 0 } });
    let draws = 0;
    node.widgets.at(-1).triggerDraw = () => { draws++; };
    node.onRemoved();
    source.widgets[0].callback();
    assert.equal(originals, 1);
    assert.equal(draws, 0);
    node.onConnectionsChange();
    draws = 0;
    node.inputs[0].link = null;
    node.onConnectionsChange();
    draws = 0;
    source.widgets[0].callback();
    assert.equal(originals, 2);
    assert.equal(draws, 0);
});


test("layout starts with prompts and places Direction/share below both characters", () => {
    const { node } = fixture();
    const order = node.widgets.map((w) => w.name);
    assert.deepEqual(order.slice(0, 2), ["base_prompt", "negative_prompt"]);
    assert.ok(order.indexOf("direction") > order.indexOf("character_2_strength"));
    assert.equal(order.indexOf("character_1_percent"), order.indexOf("direction") + 1);
    assert.equal(order.indexOf("separation"), order.indexOf("character_1_percent") + 1);
    assert.equal(node.properties.krea2_composer_widget_layout, 2);
});

test("0.3 and 0.4 saved fields are restored by their original names after positional loading", () => {
    for (const count of [13, 16, 17]) {
        const { node, get } = fixture();
        const values = { direction: "Top / Bottom", character_1_percent: 73,
            base_prompt: "scene", negative_prompt: "blurry", character_1_trigger: "a",
            character_1_prompt: "knight", character_1_lora: "a.safetensors", character_1_strength: .8,
            character_2_trigger: "b", character_2_prompt: "wizard", character_2_lora: "b.safetensors",
            character_2_strength: 1.2, separation: "Strong", boxes: "0, 0, 40, 100; 60, 0, 40, 100",
            lora_edge_softness: 32, base_strength: .75, preview_ratio: "4:5" };
        const saved = [...PREVIOUS_NAMES.slice(0, count).map((n) => values[n]), null, null];
        saved.forEach((value, i) => { if (node.widgets[i]) node.widgets[i].value = value; });
        node.onConfigure({ widgets_values: saved, properties: {} });
        for (const name of PREVIOUS_NAMES.slice(0, count)) assert.equal(get(name).value, values[name], name);
        assert.equal(node.properties.krea2_composer_widget_layout, 2);
        if (count < 17) assert.equal(get("preview_ratio").value, "Auto");
    }
});

test("reordered workflow round trip preserves prompts, LoRAs and linked field placeholders", () => {
    const first = fixture();
    for (const name of NAMES) first.get(name).value = `saved ${name}`;
    first.get("direction").value = "Free boxes";
    first.get("character_1_percent").value = 65;
    first.get("negative_prompt").value = "blurry\nwatermark\ntext";
    first.get("character_1_strength").value = .7;
    first.get("character_2_strength").value = 1.2;
    first.get("lora_edge_softness").value = 32;
    first.get("base_strength").value = 1.1;
    first.get("preview_ratio").value = "9:16";
    const info = JSON.parse(JSON.stringify({ widgets_values: NAMES.map((n) => first.get(n).value), properties: first.node.properties }));
    const second = fixture();
    info.widgets_values.forEach((value, i) => { second.node.widgets[i].value = value; });
    second.node.onConfigure(info);
    for (const name of NAMES) assert.equal(second.get(name).value, first.get(name).value, name);
    // A connected negative has no literal saved value; the marker must still
    // distinguish this layout from the old one with Direction at the top.
    info.widgets_values[1] = null;
    const third = fixture();
    third.node.onConfigure(info);
    assert.equal(third.get("base_prompt").value, first.get("base_prompt").value);
    assert.equal(third.get("direction").value, "Free boxes");
});

test("old workflows with connected split controls still use the previous order", () => {
    const { node, get } = fixture();
    const saved = PREVIOUS_NAMES.map((n) => get(n).value);
    saved[0] = null;
    saved[1] = null;
    saved[2] = "a garden";
    saved[3] = "blurry";
    node.onConfigure({ widgets_values: saved });
    assert.equal(get("base_prompt").value, "a garden");
    assert.equal(get("negative_prompt").value, "blurry");
    assert.equal(get("character_1_percent").value, null);
});

test("negative textarea remains multiline at a fixed compact height in both layouts", () => {
    const { node, get } = fixture();
    const negative = get("negative_prompt");
    const props = new Map();
    negative.element = { rows: 10, style: { setProperty: (key, value) => props.set(key, value) } };
    negative.value = "blurry\ntext\nwatermark\nextra line";
    node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value), properties: node.properties });
    assert.equal(negative.computeSize(440)[1], 64);
    assert.equal(negative.computeLayoutSize(node).maxHeight, 64);
    assert.equal(negative.options.getMinHeight(), 64);
    assert.equal(negative.options.getMaxHeight(), 64);
    assert.equal(negative.element.rows, 3);
    assert.equal(props.get("--comfy-widget-max-height"), "64px");
    assert.equal(negative.value, "blurry\ntext\nwatermark\nextra line");
    assert.equal(get("base_prompt").computeSize, undefined);
    assert.equal(get("character_1_prompt").computeSize, undefined);
});

test("compact sizing keeps a converted negative input hidden", () => {
    const { node, get } = fixture();
    const negative = get("negative_prompt");
    negative.type = "converted-widget:customtext";
    const hidden = () => [0, -4];
    negative.computeSize = hidden;
    node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value), properties: node.properties });
    assert.equal(negative.computeSize, hidden);
});

test("loading a blank boxes field restores visible selectable default boxes", () => {
    for (const value of ["", "   ", "\n\t"]) {
        const { node, get } = fixture();
        get("direction").value = "Free boxes";
        get("boxes").value = value;
        node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value), properties: node.properties });
        assert.equal(get("boxes").value, "5, 10, 42, 85; 53, 10, 42, 85");
        const { preview, frame, labels } = render(node);
        assert.equal(labels.includes("1"), true);
        assert.equal(labels.includes("2"), true);
        assert.equal(preview.mouse({ type: "pointerdown" }, [frame.x + .25 * frame.w, frame.top + .4 * frame.h], node), true);
    }
});

test("switching to Free boxes or committing an empty boxes value restores draggable frames", () => {
    const { node, get } = fixture();
    get("boxes").value = "";
    get("direction").value = "Free boxes";
    get("direction").callback();
    const expected = "5, 10, 42, 85; 53, 10, 42, 85";
    assert.equal(get("boxes").value, expected);
    get("boxes").value = "   ";
    get("boxes").callback();
    assert.equal(get("boxes").value, expected);
    const { preview, frame } = render(node);
    const point = (x, y) => [frame.x + frame.w * x / 100, frame.top + frame.h * y / 100];
    assert.equal(preview.mouse({ type: "pointerdown" }, point(25, 40), node), true);
    preview.mouse({ type: "pointermove" }, point(30, 45), node);
    preview.mouse({ type: "pointerup" }, point(30, 45), node);
    assert.match(get("boxes").value, /^10, 15, 42, 85;/);
    get("boxes").value = "";
    get("boxes").callback();
    assert.equal(preview.mouse({ type: "pointerdown" }, point(47, 95), node), true);
    preview.mouse({ type: "pointermove" }, point(42, 90), node);
    preview.mouse({ type: "pointerup" }, point(42, 90), node);
    assert.match(get("boxes").value, /^5, 10, 37, 80;/);
});

test("valid custom boxes and nonempty editing errors are preserved", () => {
    for (const text of ["10, 5, 30, 90; 50, 20, 40, 60", "10, 5, 30, 90;"]) {
        const { node, get } = fixture();
        get("boxes").value = text;
        get("direction").value = "Free boxes";
        get("direction").callback();
        node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value), properties: node.properties });
        assert.equal(get("boxes").value, text);
    }
});

test("named saved values override stale positional values after a layout change", () => {
    const { node, get } = fixture();
    const named = Object.fromEntries(NAMES.map((n) => [n, get(n).value]));
    Object.assign(named, { boxes: "10, 5, 30, 90; 50, 20, 40, 60", direction: "Free boxes",
        base_prompt: "a garden", negative_prompt: "text", character_1_prompt: "a knight", character_1_lora: "a.safetensors" });
    const stale = PREVIOUS_NAMES.map((n) => named[n]);
    stale[13] = "";
    node.onConfigure({ widgets_values: stale, widgets_values_named: named });
    for (const name of NAMES) assert.equal(get(name).value, named[name], name);
    const { labels } = render(node);
    assert.equal(labels.includes("1"), true);
    assert.equal(labels.includes("2"), true);
});

test("older named workflows lacking boxes receive defaults instead of positional leftovers", () => {
    const { node, get } = fixture();
    get("boxes").value = "";
    const named = Object.fromEntries(PREVIOUS_NAMES.slice(0, 13).map((n) => [n, get(n).value]));
    named.direction = "Free boxes";
    node.onConfigure({ widgets_values_named: named });
    assert.equal(get("boxes").value, "5, 10, 42, 85; 53, 10, 42, 85");
    assert.equal(render(node).labels.includes("1"), true);
});

test("blank connected boxes remain owned by their external text input", () => {
    const { node, get } = fixture();
    node.inputs = [{ name: "boxes", type: "STRING", link: 123 }];
    get("boxes").value = "";
    get("direction").value = "Free boxes";
    get("direction").callback();
    assert.equal(get("boxes").value, "");
    node.onConfigure({ widgets_values: NAMES.map((n) => get(n).value), properties: node.properties });
    assert.equal(get("boxes").value, "");
});
