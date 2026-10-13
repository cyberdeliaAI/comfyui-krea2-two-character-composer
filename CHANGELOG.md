# Changelog

## 0.4.5

### Metadata
- Added passive structured prompt/resource information to positive conditioning
  for Cyberdelia Image Metadata 2.2.2+. Includes the scene, exact expanded
  character prompts, original global negative, loaded regional LoRAs/strengths,
  layout and actual image dimensions.
- Data survives execution-cache reuse; the metadata reader does not need to
  execute or re-encode the Composer. Existing ports, conditioning tensors and
  regional sampling settings are unchanged. Generation has no dependency on
  the metadata extension.
- The matching zeroed negative omits this passive payload.

## 0.4.4

### Fixed
- Empty or whitespace-only Boxes values restore the two default boxes when
  creating/loading a node, entering Free boxes, or committing an empty value.
  The preview shows selectable/draggable boxes again. Valid custom coordinates
  and nonempty editing errors are preserved; connected text inputs keep ownership.
- Prefer ComfyUI widgets_values_named during restoration so stale positional
  values cannot overwrite named prompts, LoRAs or coordinates after reordering.
- Empty Boxes values in API workflows also use the default layout; the prompts
  report shows the actual coordinates used.
- Added regressions for blank values, drag recovery, named restoration, custom
  coordinates and connected inputs.

## 0.4.3

### Interface
- Moved Direction and Character 1 percent below both character groups, before
  Separation and the other layout controls. Shared prompts now come first.
- Reduced the negative prompt to a compact 64px multiline field, with scrolling
  for longer text. Other prompt sizes are unchanged. Supports canvas layout and
  the native Nodes 2.0 textarea with styles scoped to this node/field.

### Compatibility
- Restore 0.1/0.2 and 0.3–0.4.2 saved values by their original field order.
- New workflows save a layout marker; connected controls with null literal
  values are not mistaken for an older layout. Input/output names and generation
  behavior are unchanged.
- Updated the example workflow and frontend migration/layout checks.

## 0.4.2

### Performance
- Regional LoRA and LoKr projections skip token rows with zero mask weight.
  Selected rows are added back at the same positions with the same weights.
- Token selections and converted mask weights are cached across layers and steps.
  Empty unconditional masks skip adapter work; masks covering more than 75% of
  rows keep the original dense path. No approximation or threshold is applied
  to grey or feathered masks.
- Balanced avoids an unnecessary timestep GPU-to-CPU synchronization when no
  attention schedule is active. Strong and Soft retain their scheduled behavior.

### Validation
- Added dense-reference comparisons for LoRA and LoKr, float32/float16/bfloat16,
  all supported token layouts, mixed batches, noncontiguous inputs, negative
  strengths, gradients, alias safety and cache invalidation.
- Added a synthetic adapter benchmark for CPU, CUDA and MPS. It does not measure
  end-to-end generation. No Krea 2 GPU inference benchmark was available here.

## 0.4.1

### Fixed
- Base strength no longer changes an unconditional-only model call with batch size 1.
  Conditional/unconditional row roles now decide where the bias is applied, including broadcast masks.
- The preview preserves the connected latent's aspect ratio for portrait, square and landscape images.
  Width and height are fitted together inside the available space and the frame is centered.

### Tests and packaging
- Included the Python and frontend tests with regressions for both fixes and checks for the 0.4 features.
- The build tool produces both an install ZIP and a source ZIP with test/build tooling.

### Added
- A preview ratio menu with Auto and fixed 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16, 21:9, 9:21, 5:4 and 4:5 presets.
  Fixed choices work before generation and do not change the latent or generation size.
- Auto recognizes CyberKrea resolution labels, direct width/height fields, linked primitive dimensions and simple latent passthroughs.
  Unknown calculated sizes fall back to a clearly marked square preview.
- Optional width and height inputs for preview dimensions; these do not resize the input latent.

## 0.4.0

### Added
- **Free boxes** layout: two boxes anywhere in the image, dragged in the preview or typed as `x, y, w, h` in percent.
- Optional `character_1_mask` and `character_2_mask` inputs that replace a character's area.
- **Common prompts:** `{name: text}` once, `{name}` in any other field.
- **LoRA edge softness:** the two character LoRAs fade into each other at their boundary. The prompts stay separated.
- Experimental **Soft** separation: the region prompts are released for the last 40% of sampling.
- Experimental **Base strength**: how much the characters' areas follow the base prompt.
- The preview takes the image's aspect ratio from the connected empty latent.

### Changed
- The placement sentence is derived from where the two areas are, so it also works for boxes and masks.
- `prompts` output shows the layout and the options used.
- Swap with Free boxes lets the characters trade boxes.

## 0.3.1

### Fixed
- Inline `<lora:name:strength>` tags in a character's prompt load that LoRA again, as in 0.2.
  0.3.0 sent the tag to the text encoder as literal text and loaded nothing.
  Tags also work in the trigger field. Unknown or ambiguous names are now an error instead of a silent skip.
- A zero weight such as `(beard:-0.0)` no longer counts as a negative. 0.3.0 switched NegPiP on for it and then
  reported lost position data, because NegPiP correctly creates no negative positions for zero.

### Changed
- LoRA tags in the base or negative prompt raise a clear error instead of being encoded as text.
- The `prompts` output lists the LoRAs each character gets.

## 0.3.0

### Added
- Optional `negative_prompt` that works at CFG 1 through the installed NegPiP Prompt pack (comfyui-negpip-zimage).
- Per-character negatives: a negative weight such as `(beard:-0.6)` in one character's prompt.
- Trigger fields per character. Triggers go first in their own region prompt and never into the shared scene prompt.
- `negative` output: a matching empty negative for the sampler.
- `prompts` output: the scene and character prompts exactly as encoded.
- **Swap characters** button.
- Clear errors for non-Krea 2 models and when NegPiP's sidecars would reach the regional pass.

### Changed
- The scene prompt is built by the composer. Placement always follows the chosen direction and no size words are added.
  0.2.0 could say "center, dominating the frame" for a 75/25 split and added abstract "Character 1/2" labels.
- The regional engine is trimmed to what the node uses and lives in `engine/`. The upstream builder and caption helpers are gone.
- Latent channels are read from the model's latent format.
- Saved 0.1/0.2 workflows are migrated by field name when loaded.
- The split snaps to the 16px token grid, so every image token belongs to exactly one character.

### Fixed
- Combining the composer with NegPiP no longer switches the regions off silently.
- A split through the middle of a token left a column that belonged to neither character.
- The attention masks were rebuilt on every model call because ComfyUI copies the cache dict per call. They're now built once per run.

## 0.2.0

- Bundled the regional engine from ComfyUI-Krea2-Regional.

## 0.1.0

- First release.
