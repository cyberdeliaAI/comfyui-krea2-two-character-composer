"""Prompt and layout helpers. Pure Python: no torch, no ComfyUI imports."""

from __future__ import annotations

import math
import re

FREE_BOXES = "Free boxes"
DIRECTIONS = ["Left / Right", "Top / Bottom", FREE_BOXES]
SEPARATION = ["Soft", "Balanced", "Strong"]
DEFAULT_BOXES = "5, 10, 42, 85; 53, 10, 42, 85"

_LEFT_RIGHT = ("On the left side of the image", "On the right side of the image")
_TOP_BOTTOM = ("At the top of the image", "At the bottom of the image")
_HINT_LIMIT = 90
_SAME_SPOT = 0.05  # centroids closer than this give no left/right or top/bottom

# <lora:name>, <lora:name:0.8>; extra A1111-style fields like :0.8:0.5 are ignored.
_LORA_TAG = re.compile(r"<lora:([^:<>]+?)(?::\s*(-?\d*\.?\d+)\s*(?::[^<>]*)?)?>", re.IGNORECASE)
# (text:weight) with a numeric weight, the same rule as ComfyUI's parser:
# float() must accept the text after the colon, so "(x: -0.5)" is weighted
# but "(x:- 0.5)" is literal text. Nested parentheses are not supported.
_WEIGHT_GROUP = re.compile(r"\(([^()]*?):\s*(-?\d*\.?\d+)\s*\)")


def split_regions(direction, character_1_percent):
    """Return two complementary rectangles in normalized image coordinates."""
    if direction not in DIRECTIONS[:2]:
        raise ValueError("Choose Left / Right or Top / Bottom.")
    if isinstance(character_1_percent, bool):
        raise ValueError("Character 1 percentage must be between 1 and 99.")
    percent = float(character_1_percent)
    if not math.isfinite(percent) or not 1 <= percent <= 99:
        raise ValueError("Character 1 percentage must be between 1 and 99.")
    fraction = percent / 100.0
    if direction == "Left / Right":
        return (
            {"x": 0.0, "y": 0.0, "w": fraction, "h": 1.0},
            {"x": fraction, "y": 0.0, "w": 1.0 - fraction, "h": 1.0},
        )
    return (
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": fraction},
        {"x": 0.0, "y": fraction, "w": 1.0, "h": 1.0 - fraction},
    )


def parse_boxes(text):
    """'x, y, w, h; x, y, w, h' in percent of the image -> two rectangles.

    x and y are the top-left corner. Boxes may overlap (Character 1 is in
    front) and may leave space that belongs to neither character."""
    parts = [p for p in re.split(r"[;\n|]", text or "") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Free boxes needs two boxes: 'x, y, w, h; x, y, w, h' in percent.")
    rects = []
    for n, part in enumerate(parts, 1):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", part)
        if len(numbers) != 4:
            raise ValueError(f"Box {n} needs four numbers: x, y, width, height in percent.")
        x, y, w, h = (float(v) for v in numbers)
        if not (0 <= x < 100 and 0 <= y < 100 and w > 0 and h > 0
                and x + w <= 100.0001 and y + h <= 100.0001):
            raise ValueError(f"Box {n} must lie inside the image (0 to 100 percent).")
        rects.append({"x": x / 100, "y": y / 100, "w": min(w, 100 - x) / 100,
                      "h": min(h, 100 - y) / 100})
    return tuple(rects)


def relative_sides(first, second):
    """Placement phrases from two centroids (x, y in 0..1), or None.

    The axis with the larger distance decides, so two boxes side by side say
    left/right and two stacked boxes say top/bottom."""
    dx, dy = second[0] - first[0], second[1] - first[1]
    if max(abs(dx), abs(dy)) < _SAME_SPOT:
        return None
    if abs(dx) >= abs(dy):
        return _LEFT_RIGHT if dx > 0 else _LEFT_RIGHT[::-1]
    return _TOP_BOTTOM if dy > 0 else _TOP_BOTTOM[::-1]


_DEFINE = re.compile(r"\{\s*([A-Za-z][\w\- ]*?)\s*:\s*([^{}]*?)\s*\}")
_RECALL = re.compile(r"\{\s*([A-Za-z][\w\- ]*?)\s*\}")
_NUMBER = re.compile(r"-?\d*\.?\d+")


def expand_common(texts):
    """Common prompts across all fields: {key:text} defines and keeps text,
    {key} anywhere else repeats it.

    Returns (texts, unknown keys). Without any definition nothing changes.
    {a|b} choices, a number after the colon like {cat:1.2} and {word}s that
    were never defined stay as they are; unknown names are reported."""
    texts = [t or "" for t in texts]
    definitions = {}

    def is_definition(m):
        return not _NUMBER.fullmatch(m.group(2).strip())

    for text in texts:
        for m in _DEFINE.finditer(text):
            if not is_definition(m):
                continue
            key, value = m.group(1).strip().lower(), m.group(2).strip()
            if definitions.get(key, value) != value:
                raise ValueError(f"{{{m.group(1).strip()}}} is defined twice with different text.")
            definitions[key] = value
    if not definitions:
        return texts, []

    unknown = []

    def define(m):
        return m.group(2).strip() if is_definition(m) else m.group(0)

    def recall(m):
        key = m.group(1).strip().lower()
        if key in definitions:
            return definitions[key]
        if m.group(0) not in unknown:
            unknown.append(m.group(0))
        return m.group(0)

    return [_RECALL.sub(recall, _DEFINE.sub(define, text)) for text in texts], unknown


def has_negative_weights(text):
    """True when the text holds a weight below zero, like (x:-0.5).

    The numeric value decides, as in NegPiP: (x:-0.0) is not negative."""
    return any(float(m.group(2)) < 0 for m in _WEIGHT_GROUP.finditer(text or ""))


def extract_lora_tags(text):
    """Split inline <lora:name:strength> tags off a prompt.

    Returns (text without tags, [(name, strength), ...]). A missing strength
    means 1.0. Anything that starts like a tag but can't be parsed raises."""
    text = text or ""
    found = [(m.group(1).strip(), float(m.group(2)) if m.group(2) else 1.0)
             for m in _LORA_TAG.finditer(text)]
    clean = _LORA_TAG.sub("", text)
    if re.search(r"<lora:", clean, re.IGNORECASE):
        raise ValueError("Unreadable LoRA tag. Use <lora:name> or <lora:name:0.8>.")
    clean = re.sub(r"\s*,\s*(,\s*)+", ", ", clean)
    clean = re.sub(r"[ \t]+,", ",", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return clean.strip(" ,"), found


def plain_text(text):
    """Readable text without LoRA tags, weight syntax or negative groups.

    Used for the placement hints in the shared scene prompt, where negative
    concepts must never appear as positive words. Groups weighted to zero or
    below are dropped."""
    text = _LORA_TAG.sub("", text or "")
    text = _WEIGHT_GROUP.sub(lambda m: m.group(1) if float(m.group(2)) > 0 else "", text)
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def looks_like_trigger(word):
    """Single-token blobs like `xk3wchar`, `m_char` or `ohwxMan`."""
    word = word.strip()
    if not word or " " in word:
        return False
    return (any(ch.isdigit() for ch in word) or "_" in word
            or any(ch.isupper() for ch in word[1:]))


def _trigger_words(trigger):
    return [t.strip() for t in re.split(r"[,\n]", trigger or "") if t.strip()]


def region_prompt(trigger, description):
    """The prompt one region reads: trigger first, then the description.

    Trigger words the description already contains are not repeated."""
    description = (description or "").strip()
    trigger = ", ".join(
        word for word in _trigger_words(trigger)
        if not re.search(rf"(?<!\w){re.escape(word)}(?!\w)", description, re.IGNORECASE))
    if trigger and description:
        return f"{trigger}, {description}"
    return trigger or description


def hint_subject(description, trigger=""):
    """A short natural-language subject for the placement sentence.

    Trigger words never go here: the scene prompt is shared by the whole
    image, is not routed through the character LoRAs and Krea 2 tends to
    render stray trigger tokens as literal text."""
    text = plain_text(description)
    for word in _trigger_words(trigger):
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", "", text, flags=re.IGNORECASE)
    clauses = [c.strip(" .") for c in text.split(",")]
    clauses = [c for c in clauses if c and not looks_like_trigger(c)]
    if not clauses:
        return "a character"
    subject = re.sub(r"\s{2,}", " ", clauses[0])
    if len(subject) > _HINT_LIMIT:
        subject = subject[:_HINT_LIMIT].rsplit(" ", 1)[0].rstrip(" ,;:")
    return subject


def scene_prompt(base_prompt, sides, subjects):
    """Shared scene text plus one unambiguous placement sentence per character.

    `sides` comes from relative_sides() on the final region masks, so a 75/25
    split still says "left" and "right". No size words are added: the layout
    divides image area, not how large a character appears."""
    if sides:
        layout = (f"Two distinct characters in one coherent scene. "
                  f"{sides[0]}: {subjects[0]}. {sides[1]}: {subjects[1]}.")
    else:
        layout = f"Two distinct characters in one coherent scene: {subjects[0]} and {subjects[1]}."
    base = (base_prompt or "").strip().rstrip(".").strip()
    return f"{base}. {layout}" if base else layout
