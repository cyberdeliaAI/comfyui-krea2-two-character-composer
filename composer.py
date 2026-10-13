"""Krea2 Two Character Composer: two regional prompts, two character LoRAs."""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict

from . import negpip_bridge
from .prompting import (
    DEFAULT_BOXES,
    DIRECTIONS,
    FREE_BOXES,
    SEPARATION,
    expand_common,
    extract_lora_tags,
    has_negative_weights,
    hint_subject,
    parse_boxes,
    region_prompt,
    relative_sides,
    scene_prompt,
    split_regions,
)

NO_LORA = "None"
METADATA_KEY = "cyberdelia_composer_metadata"
PREVIEW_RATIOS = ["Auto", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16",
                  "21:9", "9:21", "5:4", "4:5"]
RESTRICT_END_PERCENT = 0.5  # "Strong": isolate image tokens for the first half
SOFT_RELEASE_PERCENT = 0.6  # "Soft": release prompt isolation for the last 40%
_LORA_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LORA_CACHE_MAX = 4


def latent_geometry(model):
    """(channels, downscale) from the model's latent format; Krea 2 = (16, 8)."""
    fmt = getattr(getattr(model, "model", None), "latent_format", None)
    channels = getattr(fmt, "latent_channels", 16)
    ratio = getattr(fmt, "spacial_downscale_ratio", 8)
    return int(channels), int(ratio)


def latent_dimensions(latent, channels=16, ratio=8):
    """Pixel (width, height) of a still Krea 2 latent."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    shape = getattr(samples, "shape", ())
    if len(shape) not in (4, 5) or shape[1] != channels:
        raise ValueError(
            f"Connect a Krea 2 LATENT with {channels} channels, for example from "
            "EmptySD3LatentImage. SD/SDXL latents are incompatible."
        )
    if len(shape) == 5 and shape[2] != 1:
        raise ValueError("This composer supports still images, not video latents.")
    height, width = int(shape[-2]) * ratio, int(shape[-1]) * ratio
    if height < 64 or width < 64 or height % 16 or width % 16:
        raise ValueError("Krea 2 image dimensions must be multiples of 16, at least 64px.")
    return width, height


def require_krea2(model):
    import comfy.model_base

    krea2 = getattr(comfy.model_base, "Krea2", None)
    if krea2 is None:
        raise ValueError("This ComfyUI version has no native Krea 2 support. Update ComfyUI.")
    if not isinstance(getattr(model, "model", None), krea2):
        name = type(getattr(model, "model", model)).__name__
        raise ValueError(f"Connect a Krea 2 model. The connected model is {name}.")


def _strength(value):
    strength = float(value)
    if not math.isfinite(strength) or not -10 <= strength <= 10:
        raise ValueError("LoRA strength must be finite and between -10 and 10.")
    return strength


def validate_lora(name, strength):
    """Menu selection -> (name, strength) or None; raise on bad input."""
    import folder_paths

    strength = _strength(strength)
    if name == NO_LORA or strength == 0:
        return None
    if name not in folder_paths.get_filename_list("loras"):
        raise ValueError(f"LoRA '{name}' not found. Add it to models/loras and refresh ComfyUI.")
    return name, strength


def resolve_lora_tag(name):
    """File for a <lora:name> tag: exact path, then a unique file name, then a
    unique partial match (the lookup 0.2 used). Ambiguity is an error."""
    import folder_paths

    candidates = folder_paths.get_filename_list("loras")
    norm = name.replace("\\", "/").strip().lower()
    for c in candidates:
        if c.replace("\\", "/").lower() in (norm, norm + ".safetensors"):
            return c
    stem = os.path.splitext(norm.rsplit("/", 1)[-1])[0] if norm.endswith(".safetensors") \
        else norm.rsplit("/", 1)[-1]

    def file_stem(c):
        return os.path.splitext(c.replace("\\", "/").rsplit("/", 1)[-1])[0].lower()

    for matches in ([c for c in candidates if file_stem(c) == stem],
                    [c for c in candidates if stem in file_stem(c)]):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            shown = ", ".join(matches[:4]) + (" ..." if len(matches) > 4 else "")
            raise ValueError(f"<lora:{name}> matches several LoRAs ({shown}). "
                             "Use the path shown in the LoRA menu.")
    raise ValueError(f"<lora:{name}> not found in models/loras. Check the name or use the LoRA menu.")


def character_loras(menu_row, tags):
    """Menu LoRA first, then inline tags; one entry per file, first one wins."""
    rows = [menu_row] if menu_row else []
    for name, strength in tags:
        strength = _strength(strength)
        if strength != 0:
            rows.append((resolve_lora_tag(name), strength))
    out, seen = [], {}
    for name, strength in rows:
        if name in seen:
            if seen[name] != strength:
                logging.info("[Krea2TwoCharacter] LoRA '%s' selected twice; keeping "
                             "strength %.2f, ignoring %.2f", name, seen[name], strength)
            continue
        seen[name] = strength
        out.append((name, strength))
    return out


def load_lora(name, strength):
    """State dict entry for the regional engine, with a small LRU cache."""
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("loras", name)
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        key = (path, 0, 0)
    sd = _LORA_CACHE.get(key)
    if sd is None:
        for stale in [k for k in _LORA_CACHE if k[0] == path]:
            del _LORA_CACHE[stale]
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        _LORA_CACHE[key] = sd
        while len(_LORA_CACHE) > _LORA_CACHE_MAX:
            _LORA_CACHE.popitem(last=False)
    else:
        _LORA_CACHE.move_to_end(key)
    return {"sd": sd, "strength": strength, "label": name}


def token_grid(model, ratio):
    """Pixel size of one Krea 2 image token: VAE downscale x DiT patch."""
    try:
        patch = int(getattr(model.get_model_object("diffusion_model"), "patch", 2))
    except Exception:
        patch = 2
    return ratio * max(patch, 1)


def _snap(edge, size, grid):
    """Snap an inner edge to the token grid, keeping one token on each side."""
    if edge <= 0 or edge >= 1:
        return round(edge * size)
    return min(max(round(edge * size / grid) * grid, grid), size - grid)


def rect_mask(rect, width, height, grid=1):
    """Hard (H, W) mask with inner edges on the token grid.

    The attention runs per token. An edge that cuts through a token would give
    that column 0.5 for both regions, so it would belong to neither. On the
    grid, neighbouring rectangles share one boundary and every token belongs
    to exactly one region."""
    import torch

    x0, x1 = (_snap(e, width, grid) for e in (rect["x"], rect["x"] + rect["w"]))
    y0, y1 = (_snap(e, height, grid) for e in (rect["y"], rect["y"] + rect["h"]))
    mask = torch.zeros(height, width)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def box_mask(rect, width, height, grid=1):
    """Hard (H, W) mask for a free box, every edge on the token grid and at
    least one token wide and high."""
    import torch

    def span(start, extent, size):
        a = min(max(round(start * size / grid) * grid, 0), size - grid)
        b = max(min(round((start + extent) * size / grid) * grid, size), a + grid)
        return a, b

    x0, x1 = span(rect["x"], rect["w"], width)
    y0, y1 = span(rect["y"], rect["h"], height)
    mask = torch.zeros(height, width)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def input_mask(mask, width, height):
    """A MASK input resized to the image, values 0..1. ComfyUI masks are
    (B, H, W); (H, W), (B, 1, H, W) and (B, H, W, 1) work too. Of a batch,
    the first mask is used."""
    import torch.nn.functional as F

    m = mask
    while m.ndim > 2:
        if m.shape[0] == 1 or m.ndim > 3:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            m = m[0]
    m = m.detach().float().cpu()
    if tuple(m.shape) != (height, width):
        m = F.interpolate(m[None, None], size=(height, width), mode="bilinear",
                          align_corners=False)[0, 0]
    return m.clamp(0, 1).cpu()


def centroid(mask):
    """(x, y) in 0..1 of a mask's weight."""
    import torch

    height, width = mask.shape
    total = float(mask.sum())
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    return (float((mask.sum(0) * xs).sum()) / total / width,
            float((mask.sum(1) * ys).sum()) / total / height)


def region_masks(direction, percent, boxes, mask_inputs, width, height, grid):
    """The two region masks plus a short description of the layout."""
    if direction == FREE_BOXES:
        boxes = (boxes or "").strip() or DEFAULT_BOXES
        rects = parse_boxes(boxes)
        masks = [box_mask(r, width, height, grid) for r in rects]
        layout = f"Free boxes ({boxes.strip()})"
    else:
        rects = split_regions(direction, percent)
        masks = [rect_mask(r, width, height, grid) for r in rects]
        layout = f"{direction}, {int(percent)}/{100 - int(percent)}"
    for n, mask in enumerate(mask_inputs):
        if mask is not None:
            masks[n] = input_mask(mask, width, height)
            layout += f"; Character {n + 1} from its mask input"
    check_token_coverage(masks, width, height, grid)
    return masks, layout


def check_token_coverage(masks, width, height, grid):
    """Check on the token grid, the way the engine sees the masks: each 16px
    token is averaged, a character reads its prompt where that average is
    above 0.5, and Character 1 wins ties."""
    import torch
    import torch.nn.functional as F

    size = (max(1, height // grid), max(1, width // grid))
    tokens = F.interpolate(torch.stack(masks)[:, None].float(), size=size, mode="area")[:, 0]
    first, second = tokens
    own = ((first > 0.5) & (first >= second), (second > 0.5) & (second > first))
    for n, (mask, area) in enumerate(zip(tokens, own), 1):
        if not bool((mask > 0.5).any()):
            raise ValueError(f"Character {n}'s area is empty or too thin: it covers no "
                             f"{grid}px block for more than half. Make it larger or whiter.")
        if not bool(area.any()):
            raise ValueError(f"Character {n} has no area of its own: Character 1 covers it "
                             "completely. Move or shrink one of them.")


def encode(clip, text):
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    return conditioning[0][0], dict(conditioning[0][1])


def zero_out(conditioning, drop_keys=()):
    """Like ConditioningZeroOut, minus NegPiP metadata: the result is a
    matching negative for CFG > 1 with the same token length as the positive,
    so ComfyUI can batch both without padding the regional sequence."""
    import torch

    out = []
    for cond, extra in conditioning:
        extra = {k: v for k, v in extra.items() if k not in drop_keys}
        pooled = extra.get("pooled_output")
        if pooled is not None:
            extra["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(cond), extra])
    return out


class Krea2TwoCharacterComposer:
    CATEGORY = "conditioning/krea2_regional"
    FUNCTION = "compose"
    # New outputs are appended so links in 0.1/0.2 workflows keep their slots.
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "MASK", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "latent", "region_masks", "negative", "prompts")
    OUTPUT_TOOLTIPS = (
        "Patched Krea 2 model. Connect directly to the sampler.",
        "Regional positive conditioning.",
        "The input latent, passed through.",
        "The two region masks as a batch, for previewing.",
        "Matching empty negative for the sampler (only used at CFG > 1).",
        "The scene and character prompts exactly as encoded.",
    )
    DESCRIPTION = (
        "Two regional prompts and character LoRAs for Krea 2. Split the image "
        "left/right or top/bottom, place two free boxes or connect masks. "
        "Negative prompts use the NegPiP Prompt pack when it is installed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        loras = [NO_LORA] + [n for n in folder_paths.get_filename_list("loras") if n != NO_LORA]
        text = {"default": "", "multiline": True}
        trigger = {
            "default": "", "multiline": False, "placeholder": "LoRA trigger word(s), optional",
            "tooltip": "Goes first in this character's prompt and is kept out of the "
                       "shared scene prompt, so Krea 2 doesn't render it as text.",
        }
        strength = {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}

        def character(n, side):
            return {
                f"character_{n}_trigger": ("STRING", dict(trigger)),
                f"character_{n}_prompt": ("STRING", {
                    **text, "placeholder": f"Character {n} ({side}): appearance, outfit, pose",
                    "tooltip": "Only this region reads this prompt. With NegPiP installed, "
                               "negative weights like (beard:-0.6) apply to this character only.",
                }),
                f"character_{n}_lora": (loras, {"default": NO_LORA}),
                f"character_{n}_strength": ("FLOAT", dict(strength)),
            }

        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "latent": ("LATENT",),
            "base_prompt": ("STRING", {
                **text, "placeholder": "Shared scene, background, lighting and style",
                "tooltip": "Read by the whole image. A placement sentence for both characters is added automatically.",
            }),
            "negative_prompt": ("STRING", {
                **text, "placeholder": "Optional negatives for the whole image (needs NegPiP)",
                "tooltip": "Compiled to negative weights by the installed NegPiP Prompt pack "
                           "(comfyui-negpip-zimage). Works at CFG 1. Leave empty if you don't use NegPiP.",
            }),
            **character(1, "left or top"),
            **character(2, "right or bottom"),
            "direction": (DIRECTIONS, {
                "default": DIRECTIONS[0],
                "tooltip": "Split left/right or top/bottom, or place two Free boxes. "
                           "A connected character mask replaces that character's area.",
            }),
            "character_1_percent": ("INT", {
                "default": 50, "min": 1, "max": 99, "step": 1, "display": "slider",
                "tooltip": "Image area for Character 1 (left or top). Character 2 gets 100 minus this value.",
            }),
            "separation": (SEPARATION, {
                "default": "Balanced",
                "tooltip": "Balanced preserves scene interaction. Strong also restricts "
                           "cross-region image attention for the first half of sampling. "
                           "Soft (experimental) lets the whole image read both character "
                           "prompts for the last 40% of sampling.",
            }),
        }, "optional": {
            # Optional with defaults, so 0.3 API prompts without them still run.
            "boxes": ("STRING", {
                "default": DEFAULT_BOXES, "multiline": False,
                "tooltip": "Free boxes: 'x, y, width, height; x, y, width, height' in percent "
                           "of the image, Character 1 first. Drag them in the preview. "
                           "Character 1 is in front where boxes overlap.",
            }),
            "lora_edge_softness": ("INT", {
                "default": 0, "min": 0, "max": 256, "step": 8,
                "tooltip": "Pixels over which the two character LoRAs fade into each other "
                           "at their boundary. The prompts stay separated. 0 = hard edge.",
            }),
            "base_strength": ("FLOAT", {
                "default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05,
                "tooltip": "Experimental. How strongly the characters' areas follow the base "
                           "prompt compared to their own prompt. 1.0 = unchanged.",
            }),
            "character_1_mask": ("MASK", {"tooltip": "Optional. Replaces Character 1's area. "
                                                     "Of a batch, the first mask is used."}),
            "character_2_mask": ("MASK", {"tooltip": "Optional. Replaces Character 2's area. "
                                                     "Of a batch, the first mask is used."}),
            "preview_ratio": (PREVIEW_RATIOS, {
                "default": "Auto",
                "tooltip": "Preview only: choose a fixed ratio before sampling, or Auto to read connected width/height or a recognizable latent resolution. Generation dimensions come from the LATENT.",
            }),
            "width": ("INT", {"forceInput": True,
                               "tooltip": "Optional preview width. Connect together with height. Does not resize the latent."}),
            "height": ("INT", {"forceInput": True,
                                "tooltip": "Optional preview height. Connect together with width. Does not resize the latent."}),
        }}

    def compose(self, model, clip, latent, direction, character_1_percent,
                base_prompt, negative_prompt,
                character_1_trigger, character_1_prompt, character_1_lora, character_1_strength,
                character_2_trigger, character_2_prompt, character_2_lora, character_2_strength,
                separation, boxes=DEFAULT_BOXES, lora_edge_softness=0, base_strength=1.0,
                character_1_mask=None, character_2_mask=None, preview_ratio="Auto",
                width=None, height=None):
        # Preview controls never override generation dimensions from LATENT.
        if preview_ratio not in PREVIEW_RATIOS:
            raise ValueError("Choose Auto or one of the fixed preview ratios.")
        if direction not in DIRECTIONS:
            raise ValueError("Choose Left / Right, Top / Bottom or Free boxes.")
        if separation not in SEPARATION:
            raise ValueError("Choose Soft, Balanced or Strong separation.")
        base_strength = float(base_strength)
        if not (math.isfinite(base_strength) and 0.05 <= base_strength <= 2.0):
            raise ValueError("Base strength must be between 0.05 and 2.")
        if not 0 <= int(lora_edge_softness) <= 256:
            raise ValueError("LoRA edge softness must be between 0 and 256 pixels.")
        require_krea2(model)
        channels, ratio = latent_geometry(model)
        width, height = latent_dimensions(latent, channels, ratio)
        grid = token_grid(model, ratio)
        masks, layout = region_masks(direction, character_1_percent, boxes,
                                     (character_1_mask, character_2_mask), width, height, grid)

        (base_prompt, negative_prompt, character_1_trigger, character_1_prompt,
         character_2_trigger, character_2_prompt), unknown_common = expand_common(
            [base_prompt, negative_prompt, character_1_trigger, character_1_prompt,
             character_2_trigger, character_2_prompt])

        base_prompt, base_tags = extract_lora_tags(base_prompt)
        negative_prompt, negative_tags = extract_lora_tags(negative_prompt)
        if base_tags or negative_tags:
            raise ValueError(
                "LoRA tags only work in a character's trigger or prompt. For a LoRA on the "
                "whole image, use a LoraLoader before the composer.")

        characters, character_tags = [], []
        for trigger, description in ((character_1_trigger, character_1_prompt),
                                     (character_2_trigger, character_2_prompt)):
            trigger, trigger_tags = extract_lora_tags(trigger)
            description, description_tags = extract_lora_tags(description)
            characters.append((trigger, description))
            character_tags.append(trigger_tags + description_tags)
        region_texts = [region_prompt(t, d) for t, d in characters]
        for n, region_text in enumerate(region_texts, 1):
            if not region_text:
                raise ValueError(f"Enter a trigger word or a prompt for Character {n}.")
        lora_rows = (
            character_loras(validate_lora(character_1_lora, character_1_strength), character_tags[0]),
            character_loras(validate_lora(character_2_lora, character_2_strength), character_tags[1]),
        )

        scene = scene_prompt(base_prompt, relative_sides(*(centroid(m) for m in masks)),
                             [hint_subject(d, t) for t, d in characters])
        negative = (negative_prompt or "").strip()
        base_text = scene
        # Which prompts need NegPiP? Only those get its CLIP: the stock Krea 2
        # tokenizer ignores weight syntax, NegPiP's enables it. A negative in
        # one prompt must not change how the other prompts are encoded.
        needs_negpip = [bool(negative) or has_negative_weights(base_prompt or "")]
        needs_negpip += [has_negative_weights(t) for t in region_texts]

        negpip = None
        negpip_clip = None
        if any(needs_negpip):
            negpip = negpip_bridge.require()
            # NegPiP puts its wrapper first; the regional wrapper added later
            # runs inside it and sees the sequence without sidecars.
            model, negpip_clip = negpip.patch(model, clip)
            if negative:
                base_text = negpip.merge(scene, negative)

        chunks = []
        for text, use_negpip in zip([base_text] + region_texts, needs_negpip):
            if use_negpip:
                cond, extra = encode(negpip_clip, text)
                sidecar = negpip.sidecar_tokens(cond, extra, has_negative_weights(text))
            else:
                cond, extra = encode(clip, text)
                sidecar = 0
            chunks.append((cond, extra, sidecar))

        drop = ("attention_mask",) + (negpip.metadata_keys if negpip else ())
        base_cond, base_extra, base_sidecar = chunks[0]
        # Krea 2 ignores text attention masks, and a base-length mask or stale
        # NegPiP metadata would not match the combined sequence.
        base_extra = {k: v for k, v in base_extra.items() if k not in drop}

        regions = [
            {"cond": cond, "sidecar": sidecar, "mask": mask,
             "loras": [load_lora(*row) for row in rows]}
            for (cond, _, sidecar), mask, rows in zip(chunks[1:], masks, lora_rows)
        ]

        from .engine.regional import apply_regional

        patched, positive = apply_regional(
            model,
            {"cond": base_cond, "extra": base_extra, "sidecar": base_sidecar},
            regions,
            restrict_img_attn=separation == "Strong",
            restrict_end_percent=RESTRICT_END_PERCENT,
            release_text_percent=SOFT_RELEASE_PERCENT if separation == "Soft" else None,
            exclusive_masks=True,
            unmaskable_layers="skip",
            base_strength=base_strength,
            lora_feather_px=int(lora_edge_softness),
        )

        # Passive text/resource information travels with cached conditioning.
        # It does not add tokens, weights, model options or output sockets.
        metadata = {
            "version": 1,
            "scene_prompt": scene,
            "encoded_scene_prompt": base_text,
            "character_prompts": list(region_texts),
            "negative_prompt": negative,
            "loras": [[{"name": name, "strength": float(strength)} for name, strength in rows]
                      for rows in lora_rows],
            "layout": layout,
            "separation": separation,
            "lora_edge_softness": int(lora_edge_softness),
            "base_strength": base_strength,
            "width": width, "height": height,
            "negpip": bool(negpip),
        }
        positive = [[cond, {**extra, METADATA_KEY: metadata}] for cond, extra in positive]

        import torch

        def lora_note(rows):
            return ", ".join(f"{name} @ {strength:g}" for name, strength in rows) or "none"

        report = "\n".join([
            f"Scene: {base_text}",
            f"Character 1: {region_texts[0]}",
            f"Character 1 LoRAs: {lora_note(lora_rows[0])}",
            f"Character 2: {region_texts[1]}",
            f"Character 2 LoRAs: {lora_note(lora_rows[1])}",
            f"Layout: {layout}",
            f"Separation: {separation}; LoRA edge softness: {int(lora_edge_softness)}px; "
            f"base strength: {base_strength:g}",
            f"NegPiP: {'on' if negpip else 'off'}",
        ] + ([f"Common prompts not defined, left as text: {', '.join(unknown_common)}"]
             if unknown_common else []))
        if unknown_common:
            logging.warning("[Krea2TwoCharacter] common prompts not defined: %s",
                            ", ".join(unknown_common))
        return (patched, positive, latent, torch.stack(masks),
                zero_out(positive, drop + (METADATA_KEY,)), report)
