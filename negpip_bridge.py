"""Optional bridge to an installed NegPiP Prompt pack (comfyui-negpip-zimage).

No NegPiP code is bundled or copied here: the composer calls the installed
package at runtime, the same way it calls ComfyUI itself. Without the pack,
everything except negative prompts keeps working.

How the two patches coexist:

* NegPiP's CLIP patch appends "sidecar" tokens to the END of each encoded
  prompt that holds negative weights. The sidecar stores the negative token
  positions relative to that prompt.
* NegPiP's diffusion wrapper always sits first (outermost). It strips every
  sidecar, rebuilds absolute positions per prompt segment and flips the
  attention values of those tokens.
* The regional wrapper runs inside it, so it lays out its segments on the
  stripped sequence (see engine.regional.layout_chunks).
"""

from __future__ import annotations

import importlib
import sys

PACK = "comfyui-negpip-zimage"
_POSITIONS_KEY = "krea2_negpip_negative_positions"
_SOURCE_LENGTH_KEY = "krea2_negpip_source_length"
_SIDECAR_TOKENS_KEY = "krea2_negpip_sidecar_tokens"


class NegPipUnavailable(RuntimeError):
    pass


class NegPip:
    def __init__(self, merge_prompts, krea2_module):
        self._merge = merge_prompts
        self._krea2 = krea2_module
        self.positions_key = getattr(krea2_module, "NEGATIVE_POSITIONS_EXTRA_KEY", _POSITIONS_KEY)
        self.source_length_key = getattr(krea2_module, "NEGATIVE_SOURCE_LENGTH_EXTRA_KEY", _SOURCE_LENGTH_KEY)
        self.sidecar_tokens_key = getattr(krea2_module, "NEGATIVE_SIDECAR_TOKENS_EXTRA_KEY", _SIDECAR_TOKENS_KEY)

    @property
    def metadata_keys(self):
        return (self.positions_key, self.source_length_key, self.sidecar_tokens_key)

    def patch(self, model, clip):
        """Return NegPiP-patched clones of model and CLIP."""
        return self._krea2.patch_krea2_negpip(model, clip)

    def merge(self, positive, negative):
        """Merge a negative prompt into the positive one as negative weights."""
        return self._merge(positive, negative)

    def sidecar_tokens(self, cond, extra, expect_negative):
        """Number of sidecar tokens at the end of one encoded prompt."""
        positions = extra.get(self.positions_key)
        if not positions:
            if expect_negative:
                raise RuntimeError(
                    "NegPiP compiled a negative weight but its position data was "
                    "lost while encoding. Load the text encoder with CLIPLoader "
                    "type 'krea2' and update comfyui-negpip-zimage.")
            return 0
        sidecar = int(extra.get(self.sidecar_tokens_key) or 0)
        source = extra.get(self.source_length_key)
        if sidecar <= 0 or (source is not None and int(source) + sidecar != int(cond.shape[1])):
            raise RuntimeError(
                "Unexpected NegPiP conditioning layout. Update both "
                "comfyui-negpip-zimage and this composer.")
        return sidecar


def find():
    """Locate the installed NegPiP pack among the loaded custom nodes."""
    for name, module in list(sys.modules.items()):
        if not name.endswith(".src.nodes"):
            continue
        if not (hasattr(module, "NegPipPrompt") and hasattr(module, "merge_prompts")):
            continue
        package = name[: -len(".nodes")]
        try:
            krea2 = importlib.import_module(package + ".negpip_krea2")
        except Exception as exc:  # older pack or incomplete install
            raise NegPipUnavailable(
                f"Found {PACK}, but its Krea 2 module failed to load: {exc}") from exc
        if not hasattr(krea2, "patch_krea2_negpip"):
            raise NegPipUnavailable(f"Your {PACK} version has no Krea 2 support. Update it.")
        return NegPip(module.merge_prompts, krea2)
    return None


def require():
    negpip = find()
    if negpip is None:
        raise NegPipUnavailable(
            f"Negative prompts need the NegPiP Prompt pack ({PACK}). Install it "
            "from the ComfyUI Manager or github.com/cyberdeliaAI/comfyui-negpip-zimage "
            "and restart ComfyUI, or clear the negative fields.")
    return negpip
