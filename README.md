# Krea2 Two Character Composer

<p align="center">
  <img src="assets/icon.png" alt="Krea2 Two Character Composer icon" width="200">
</p>

[![Tests](https://github.com/cyberdeliaAI/comfyui-krea2-two-character-composer/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberdeliaAI/comfyui-krea2-two-character-composer/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A single ComfyUI node that puts **two characters, each with their own LoRA**, into one Krea 2 image.
Split the image left/right or top/bottom, drag two free boxes, or connect your own masks.
The node shows a live layout preview and relabels its fields to match the chosen layout.

Optional **negative prompts** work at CFG 1 through the
[NegPiP Prompt](https://github.com/cyberdeliaAI/comfyui-negpip-zimage) pack.

**[Download 0.4.5](https://github.com/cyberdeliaAI/comfyui-krea2-two-character-composer/releases/tag/v0.4.5)** ·
**[Example workflow](example_workflows/two_characters.json)**

The regional engine is included. You do not need to install ComfyUI-Krea2-Regional separately.
This release supports Krea 2; Z-Image Turbo needs a different composer.

## First image: two characters, 60/40

1. Install the node and open your working Krea 2 workflow, or import the example below.
2. Connect MODEL, CLIP and the empty LATENT to the composer. Connect its model,
   positive, negative and latent outputs to the sampler as shown in [Wiring](#wiring).
3. Enter the shared setting in **Base prompt**, for example `Two adults in a garden, daylight, illustration`.
4. Select Character 1's LoRA, enter its required trigger and describe its appearance, outfit and pose.
   Repeat for Character 2 using its own LoRA and trigger.
5. Choose **Left / Right**, set **Character 1 percent** to `60`, and start with
   **Balanced**, **LoRA edge softness** `0` and **Base strength** `1.0`.
6. Generate using the sampler settings that already work for your Krea 2 model.

Character 1 gets the left 60% of the image and Character 2 the right 40%.
Choose **Top / Bottom** for the same division vertically. The percentages control
the regions; describe the intended character sizes in the prompts.

## Installation

1. Use a recent ComfyUI with native Krea 2 support (the `krea2` CLIPLoader type).
2. Unzip `comfyui-krea2-two-character-composer-0.4.5.zip` into `ComfyUI/custom_nodes`, or clone the repository there.
   The folder should look like this, without an extra nested folder:

   ```text
   ComfyUI/custom_nodes/
   └── comfyui-krea2-two-character-composer/
       ├── __init__.py
       ├── composer.py
       ├── prompting.py
       ├── negpip_bridge.py
       ├── engine/
       └── web/composer.js
   ```

3. Put your **Krea 2** character LoRAs in `ComfyUI/models/loras`.
4. Restart ComfyUI and refresh the browser. Search for **Krea2 Two Character Composer**.

To install with Git, run this inside `ComfyUI/custom_nodes`:

```sh
git clone https://github.com/cyberdeliaAI/comfyui-krea2-two-character-composer.git
```

No extra Python packages are needed. Nothing is downloaded at runtime.

**Negative prompts (optional):** also install
[comfyui-negpip-zimage](https://github.com/cyberdeliaAI/comfyui-negpip-zimage) 2.1 or newer.
Without it the node works normally; only the negative features ask you to install it.

## Wiring

```text
Krea 2 MODEL ──┐
Krea 2 CLIP ───┼── Two Character Composer ─ model ──── KSampler.model
Krea 2 LATENT ─┘                          ├ positive ─ KSampler.positive
                                          ├ negative ─ KSampler.negative
                                          └ latent ─── KSampler.latent_image
KSampler → VAE Decode → Save Image
```

- Use **EmptySD3LatentImage** for the latent. Despite the name it makes the 16-channel latent Krea 2 uses.
- Load the text encoder with **CLIPLoader** type `krea2`.
- Connect the composer's `model` output **directly** to the sampler.
- `negative` is a matching empty negative. It only matters at CFG > 1, but always connecting it is the safe default.
- `prompts` shows exactly what was encoded. Connect it to **Preview Any** when you're tuning prompts.
- `region_masks` is optional: **Mask To Image → Preview Image** shows both masks as a batch.
- `character_1_mask` and `character_2_mask` are optional inputs. See [Layouts](#layouts).

Drag [the example workflow](example_workflows/two_characters.json) into ComfyUI and pick your own model, encoder, VAE and LoRAs.
The sampler starts at **8 steps / CFG 1** for Turbo. For RAW, use the settings of your working RAW workflow.
The composer doesn't change sampler or model-sampling settings.

## Node layout

Fill the shared scene and compact multiline negative prompt first, followed by
Character 1 and Character 2. **Direction** and the percentage slider sit below
both characters, together with the other region options and layout preview.
The negative prompt keeps roughly three visible lines; longer text can be
scrolled. Other prompt fields keep their usual size.

Saved workflows with named values restore by those names; older positional-only
0.2–0.4.2 workflows use their original field order. Workflows
saved with the new order carry a layout marker so subsequent loads retain the
same values, including when fields are converted to inputs. API field names
and connections are unchanged.

## Fields

| Field | Meaning |
|---|---|
| Direction | `Left / Right`, `Top / Bottom` or `Free boxes`. In a split, Character 1 is left or top. |
| Character 1 percent | 1–99. Character 2 gets `100 − percent`. This divides image **area**, not how large a character appears. The boundary snaps to Krea 2's 16px token grid. Not used with Free boxes. |
| Base prompt | Shared scene, background, lighting and style. Read by the whole image. |
| Negative prompt | Optional. Negatives for the whole image, compiled by NegPiP. |
| Character N trigger | The LoRA trigger word(s). Optional. |
| Character N prompt | Appearance, outfit and pose. Only this character's region reads it. |
| Character N LoRA / strength | A LoRA from `models/loras`, or `None`. Strength 0 also turns it off. |
| Separation | `Balanced` keeps the scene coherent. `Strong` also blocks image attention between the regions for the first half of sampling. `Soft` is experimental, see below. |
| Boxes | The two free boxes: `x, y, width, height; x, y, width, height` in percent. Only used with Free boxes. |
| LoRA edge softness | Pixels over which the two character LoRAs fade into each other. `0` = hard edge. |
| Base strength | Experimental, see below. `1.0` = unchanged. |
| Preview ratio | `Auto` or a fixed ratio, such as `4:5`, `16:9` or `9:16`. Changes the layout preview only. |

**Swap characters** exchanges the trigger, prompt, LoRA selection and LoRA strength
between the two character fields. With **Free boxes**, the box coordinates stay
fixed: each figure takes the location and size of the other box. The blue `1` and
pink `2` label the regions, not permanent identities. For example, if blue 1 is a
woman and pink 2 a man, after swapping blue 1 is the man and pink 2 the woman.
In a split, the percentage is also inverted, so a 60/40 split becomes 40/60 and
each figure retains its share while changing sides. Connected mask inputs and
other input links are not swapped; exchange those links manually when needed.

**Inline LoRA tags:** `<lora:name>` or `<lora:name:0.8>` in a character's trigger or prompt adds a LoRA for that character only,
on top of the menu. The tag is removed before encoding. `name` can be the path from the LoRA menu, the file name or a unique part of it.
If the same file is also in the menu, the menu's strength wins. Tags in the base or negative prompt are rejected:
load a LoRA for the whole image with a LoraLoader before the composer.

## Layouts

- **Split:** `Left / Right` or `Top / Bottom` with Character 1's percentage. Every image token belongs to exactly one character.
- **Free boxes:** two boxes anywhere in the image. Drag a box in the preview to move it and drag a corner to resize it,
  or type the numbers in the `boxes` field. An empty field restores the two default boxes
  when loading a workflow or choosing Free boxes. Valid custom coordinates are preserved. Edges snap to the 16px token grid.
  Boxes may overlap; Character 1 owns the prompt region in the overlap. This is
  region priority, not a guarantee that the model draws that person in front.
  Describe the intended foreground/background relationship in your prompts.
  Space outside both boxes belongs to neither character and only follows the base prompt.
  The preview takes its shape from the connected empty latent when it can read the size.
- **Masks:** connect a MASK to `character_1_mask` or `character_2_mask`, for example from the mask editor of a Load Image node.
  It replaces that character's area in any layout, and the mask is resized to the image.
  The character's prompt applies where the mask is more than half white. Grey also scales its LoRA down there.
  Of a mask batch, the first mask is used.

For example, `5, 10, 40, 55; 35, 5, 60, 85` puts a smaller first box on the left
and a larger second box toward the right, with an overlap. The coordinates are
`x` from the left, `y` from the top, then width and height, all in percent.
Keep each box inside the image (`x + width <= 100`, `y + height <= 100`).

**LoRA edge softness does not change the boxes or the overlap priority.** It
softens only the character LoRA influence near the edges; prompts keep their
region boundaries. Start at `0` for the strongest identity separation. Try
`32` and increase cautiously if a boundary looks abrupt: higher values allow
more LoRA influence near the other character and can blend appearance details.

The node stops with a clear message when a character ends up without a 16px block of its own (an empty, very thin or faint mask),
or when Character 1 covers Character 2 completely.

## Preview proportions before generation

Choose **Preview ratio** to match your resolution selector. Available presets:
`1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`, `21:9`, `9:21`, `5:4`, `4:5`.
For example, choose **4:5** for **1152×1440**. The preview updates immediately,
including in Nodes 2.0. Free-box coordinates remain percentages of the image.

**Auto** reads visible width/height widgets, CyberKrea resolution labels and simple
latent passthroughs. Optional `width` and `height` INT inputs let you connect both
outputs from a resolution node, or primitive dimensions. Visible source changes
refresh the preview. Values that are calculated only when the graph runs cannot
be predicted; Auto shows a square with “size unavailable” in that case. Choose a
fixed preset when using those nodes.

The preset and optional dimension inputs control the preview only. Generation
and region masks always use the actual LATENT dimensions, so keep your chosen
preview ratio consistent with your latent source. A preset never resizes the latent.

## Common prompts

Write a shared piece of prompt once and reuse it in other fields:

```text
Base prompt:   A classroom at noon, {outfit: navy school uniform, red tie}.
Character 1:   a tall boy with black hair, {outfit}
Character 2:   a girl with a blonde ponytail, {outfit}
```

`{name: text}` defines it and keeps the text where it stands. `{name}` in any field, before or after the definition, repeats it.
Names aren't case sensitive. Defining one name twice with different text stops the node with a message.
A name that's never defined stays as text and is listed in the `prompts` output.
`{a|b}` choices and `{word:1.2}` stay as they are, and prompts without any definition are never changed.

## Experimental options

These need testing on real images. Their defaults change nothing.

- **Soft:** the region prompts are separated while the layout forms. For the last 40% of sampling every part of the image may read both character prompts, which can help the two blend into one scene.
  Character LoRAs stay in their own regions.
- **Base strength:** how much the characters' areas follow the base prompt compared to their own prompt.
  Below `1.0` the character prompts weigh more; above `1.0` the shared scene and style weigh more. Try `0.5` to `1.5`.
- **LoRA edge softness:** not experimental in how it works, but how much looks good depends on your LoRAs. Try `32` to `96`.

## What the model receives

The composer builds the shared scene prompt itself and adds one placement sentence per character:

```text
Two people in a garden, daylight, illustration. Two distinct characters in one coherent scene.
On the left side of the image: a man with spiky black hair. On the right side of the image: a monk.
```

- Positions come from where the two areas actually are: left/right or top/bottom, whichever axis separates them most.
  A 75/25 split still says "left" and "right". When both areas share the same centre, no positions are named.
- No size words are added.
- **Trigger words never enter the scene prompt.** It isn't routed through the character LoRAs, and Krea 2 tends to render stray trigger tokens as text.
  Each trigger goes first in its own character prompt instead.
- The placement sentence uses the first clause of each character prompt, without weights, negatives or `<lora:…>` tags.

## Negatives with NegPiP

Krea 2 Turbo runs at CFG 1, where a normal negative prompt does nothing.
NegPiP turns negatives into negative weights inside the positive prompt, so they work at CFG 1.

- **Whole image:** fill `Negative prompt`, e.g. `blurry, (text, watermark:0.5)`.
- **One character only:** write a negative weight in that character's prompt, e.g. `a monk, (beard:-0.6)`.
  Only that character's region reads it.

The composer finds the installed NegPiP pack and calls it. Don't put a separate NegPiP Prompt node in front of it.
Only prompts that hold a negative go through NegPiP. Prompts without negatives encode exactly as before.
Note that NegPiP turns on weight syntax like `(red hair:1.3)` in the prompts it encodes. The stock Krea 2 encoder reads weights as plain text.

Under the hood: NegPiP appends small "sidecar" tokens to each prompt that holds negatives, and its model patch strips them before the model runs.
The composer lays out its regions on that stripped sequence. Without this, the regions would silently stop working.

## Behavior and limits

- Character LoRAs are applied per region in the same pass. LoRAs you load globally **before** the composer stay global.
- Masks stay fixed during sampling. Where two areas overlap, Character 1 wins.
- Some LoRA layers (timestep embedders from newer trainers) have no spatial axis. They're skipped for character LoRAs so they can't affect the whole image.
- Regions guide the model but don't guarantee perfect identity or placement, especially with overlapping people, hands across the boundary or very narrow regions.
  `Strong` separates harder but can reduce coherence.
- Still images only, including batches. No video and no Krea 2 reference latents.
- Krea 2 only. Other models are rejected with a clear message.

## Saved prompt metadata

Use **Save Image With Metadata (Cyberdelia)** from
[comfyui-cyberdelia-metadata](https://github.com/cyberdeliaAI/comfyui-cyberdelia-metadata) **2.2.2 or newer** to retain the scene, both character
prompts with triggers and the original global negative prompt. The positive
metadata labels the three parts as Scene, Character 1 and Character 2.
Common-prompt references are already expanded and inline LoRA tags are removed
from the encoded texts; loaded regional LoRAs and strengths are recorded separately.

No additional connection from the `prompts` output is needed. The composer
attaches passive text/resource information to its cached positive conditioning,
so it also survives ComfyUI cache hits. Existing ports and sampling tensors are
unchanged. The metadata extension is optional; generation works without it.

## Performance

Version 0.4.2 skips LoRA calculations for tokens where that character's mask is
zero. The selections and converted mask weights are reused across layers and
sampling steps. Both character LoRAs keep their existing region boundaries and
strengths; grey values and edge softness keep the same weighting. Masks that
cover most of a layer's tokens use the original dense calculation.

This reduces the work done by the character adapters. The base model, combined
prompt sequence and regional attention still need to run, so generation remains
heavier than a workflow without regional conditioning. Actual speed depends on
the GPU, LoRA rank, region coverage and VRAM. Floating-point rounding can differ
from 0.4.1; identical seeds are not a promise of pixel-identical images.

`Balanced` also avoids reading a GPU timestep back to the CPU when no region
schedule is active. `Strong` and `Soft` keep their original schedules.

The source ZIP includes a synthetic comparison tool:

```sh
python tools/benchmark_lora.py --device cuda --dtype float16
```

This compares the old and optimized **adapter calculations only**, with matching
masks and weights. It excludes the base model, attention, encoding and decoding;
it does not measure total image generation time. CUDA inference quality and
end-to-end generation speed still need to be checked on your actual workflow.

## Upgrading

Replace the folder. Saved 0.2, 0.3 and 0.4 workflows load with their values in the right fields, and the new options start at their defaults.

For a Git installation, run `git pull` inside the extension folder, restart
ComfyUI and refresh the browser. Avoid keeping two copies of this extension in
`custom_nodes`.

From 0.2.x:
The new trigger and negative fields start empty, so your old prompts with inline triggers and `<lora:…>` tags keep working.
Output slots are unchanged; `negative` and `prompts` were added after them.
You can replace a separate **ConditioningZeroOut** with the new `negative` output.

## Troubleshooting

| Problem | Check |
|---|---|
| Node missing after installation | Check for a nested ZIP folder, restart ComfyUI, and read the startup log. Update ComfyUI if it lacks native Krea 2 support. |
| LoRA missing from the menu | Put a Krea 2 LoRA in `models/loras` and refresh ComfyUI's model list or restart. |
| Character identities mix | Keep character descriptions and triggers in their own fields, use edge softness `0`, and try Strong. A character LoRA loaded before the composer affects the whole image. |
| Free boxes cannot be dragged | Clear the Boxes field to restore defaults. If it contains text, use two groups of four numbers separated by `;`. Connected masks override boxes during generation. |
| Preview has the wrong proportions | Select a matching Preview ratio. Auto cannot read dimensions computed only at execution time. Generation always uses the input LATENT. |
| Negative prompt requests NegPiP | Install the linked NegPiP pack, restart ComfyUI, and keep it installed. A separate NegPiP node is not needed. |
| Thin mask or fully covered character error | Enlarge or brighten the mask, or move overlapping boxes so each character has its own area on the 16px grid. |
| Prompts missing from saved image metadata | Update Cyberdelia Image Metadata to 2.2.2+ and use its Save Image With Metadata node. No connection from `prompts` is needed. |

When reporting an issue, include the Composer and ComfyUI versions, the error
message and a minimal workflow with private prompts and file paths removed.

## Tests

```sh
python3 -m unittest discover -s . -p "test_*.py" -v
node --test tests/frontend.test.mjs
```

The tests need PyTorch and run the real regional engine with stand-ins for ComfyUI.
Two optional environment variables switch in real code:

- `COMFYUI_PATH=/path/to/ComfyUI` uses ComfyUI's own wrapper stacking.
- `NEGPIP_PATH=/path/to/comfyui-negpip-zimage` runs the NegPiP integration tests against the real NegPiP patch, sidecar format and wrapper.

These are not GPU inference tests. Image quality still needs checking in ComfyUI with real Krea 2 LoRAs.

## Credits and license

MIT license, © 2026 Wilfred Helmig.

The regional attention engine in `engine/` is derived from
[ComfyUI-Krea2-Regional](https://github.com/januspluto/ComfyUI-Krea2-Regional) by januspluto (MIT).
See [engine/NOTICE.md](engine/NOTICE.md) and [engine/LICENSE-krea2-regional](engine/LICENSE-krea2-regional).

NegPiP support calls the separately installed
[comfyui-negpip-zimage](https://github.com/cyberdeliaAI/comfyui-negpip-zimage) pack (AGPL-3.0). None of its code is included here.

Not affiliated with Krea.
