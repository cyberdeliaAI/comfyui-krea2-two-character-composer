"""Single-pass regional attention and per-region LoRA routing for Krea 2.

Derived from ComfyUI-Krea2-Regional by januspluto
(https://github.com/januspluto/ComfyUI-Krea2-Regional, commit
307081f2b954d9f5e683dadafdb53ca971ebdbfd), MIT license, see
engine/LICENSE-krea2-regional. Trimmed to what the composer uses:

* All prompts (base + regions) are concatenated into one Krea 2 text
  sequence. Krea 2 is a single-stream DiT, so text and image tokens share
  every attention op. A joint attention mask injected through
  ``optimized_attention_override`` lets each region's image tokens read only
  their own prompt plus the shared base prompt.
* LoRAs are applied per token: eligible Linears inside the DiT get a class
  swap that adds ``mask * scale * (B A x)`` on top of the base projection,
  where the mask covers the region's image tokens and its prompt tokens.

Changes from upstream: no node classes, no adaptive masks, no Region Lock,
own wrapper/bundle keys (so it never collides with an installed upstream
copy) and NegPiP sidecar awareness. Chunks may carry trailing NegPiP sidecar
tokens; the sampler sees them, but the NegPiP wrapper (always outermost)
strips them before this wrapper runs, so segments are laid out on the
stripped sequence.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import re

import torch
import torch.nn.functional as F

from comfy.patcher_extension import WrappersMP

BUNDLE_KEY = "krea2_two_character_regional"
CONSUMED_KEY = "krea2_two_character_regional_consumed"
LOG = "[Krea2TwoCharacter]"


class _Ctx:
    """Token masks for the *current* diffusion-model call. Adapters only
    contribute when their uid is present, so any call where the wrapper did
    not populate this (negative prompt, other models) runs clean."""

    def __init__(self):
        self.token_masks: dict[str, torch.Tensor] = {}  # uid -> (B, L, 1)
        self.token_plans = {}
        self.txtlen = 0
        self.imglen = 0
        self.txtlayers = -1

    def clear(self):
        self.token_masks = {}
        self.token_plans = {}
        self.txtlen = 0
        self.imglen = 0
        self.txtlayers = -1


CTX = _Ctx()
_UID = itertools.count()


class _RunCache:
    """Mutable holder for per-run mask caches. ComfyUI's merge of
    transformer_options copies dicts and lists but passes other objects by
    reference, so this survives from one model call to the next."""

    __slots__ = ("rt",)

    def __init__(self):
        self.rt = None


# ---------------------------------------------------------------------------
# LoRA checkpoint parsing (kohya / peft / diffusers key styles, LoKr)
# ---------------------------------------------------------------------------

_PREFIXES = (
    "diffusion_model.",
    "transformer.",
    "model.",
    "lora_unet_",
    "lora_transformer_",
)


def _normalize_lora_sd(sd: dict) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for k, v in sd.items():
        m = re.match(r"^(.*?)\.(lora_A|lora_down)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["down"] = v
            continue
        m = re.match(r"^(.*?)\.(lora_B|lora_up)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["up"] = v
            continue
        m = re.match(r"^(.*?)\.(lokr_w1|lokr_w1_a|lokr_w1_b|lokr_w2"
                     r"|lokr_w2_a|lokr_w2_b|lokr_t2)$", k)
        if m:
            groups.setdefault(m.group(1), {})[m.group(2)] = v
            continue
        m = re.match(r"^(.*?)\.alpha$", k)
        if m:
            groups.setdefault(m.group(1), {})["alpha"] = float(v)
            continue
    out = {}
    for key, g in groups.items():
        orig_key = key
        entry = None
        if "down" in g and "up" in g:
            entry = {"type": "lora", "down": g["down"], "up": g["up"]}
            if "alpha" in g:
                entry["alpha"] = g["alpha"]
        elif ("lokr_w1" in g or "lokr_w1_a" in g) and \
                ("lokr_w2" in g or "lokr_w2_a" in g):
            if "lokr_t2" in g:
                logging.warning("[Krea2TwoCharacter] '%s': tucker LoKr is not "
                                "supported for these linear layers — skipped.",
                                key)
                continue
            dim = None
            if "lokr_w1" in g:
                w1 = g["lokr_w1"].float()
            else:
                dim = g["lokr_w1_b"].shape[0]
                w1 = (g["lokr_w1_a"].float() @ g["lokr_w1_b"].float())
            if "lokr_w2" in g:
                w2 = g["lokr_w2"].float()
            else:
                dim = g["lokr_w2_b"].shape[0]
                w2 = (g["lokr_w2_a"].float() @ g["lokr_w2_b"].float())
            if w1.ndim != 2 or w2.ndim != 2:
                logging.warning("[Krea2TwoCharacter] '%s': non-2D LoKr factors "
                                "(conv?) — skipped.", key)
                continue
            alpha_scale = (g["alpha"] / dim
                           if ("alpha" in g and dim) else 1.0)
            entry = {"type": "lokr", "w1": w1, "w2": w2,
                     "alpha_scale": alpha_scale}
        if entry is None:
            continue
        entry["orig"] = orig_key
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                break
        out[key] = entry
    return out


# ---------------------------------------------------------------------------
# Token-masked LoRA via class swap (paths / params / state_dict unchanged)
# ---------------------------------------------------------------------------


def _adapt_mask(m, x):
    """Fit a full-sequence (B, L, 1) token mask to whatever the current layer
    sees: the joint sequence (DiT blocks), text-only (txtfusion refiners /
    txtmlp), image-only (patch embed), per-token rows (txtfusion layerwise
    blocks, where the batch dim is batch*txtlen), or the 4D projector."""
    if x.ndim == 3:
        s = x.shape[1]
        if s == m.shape[1]:
            mm = m
        elif s == CTX.txtlen:
            mm = m[:, :s]
        elif s == CTX.imglen:
            mm = m[:, CTX.txtlen:CTX.txtlen + s]
        elif (s == CTX.txtlayers and CTX.txtlen > 0
              and x.shape[0] % CTX.txtlen == 0):
            vec = m[:, :CTX.txtlen, 0]            # (Bm, txtlen)
            reps = x.shape[0] // CTX.txtlen
            if vec.shape[0] == 1:
                flat = vec[0].repeat(reps)
            elif vec.shape[0] == reps:
                flat = vec.reshape(-1)
            else:
                return None
            return flat[:, None, None]            # per-row gate
        else:
            return None
        return mm if mm.shape[0] in (1, x.shape[0]) else None
    if x.ndim == 4 and x.shape[1] == CTX.txtlen:  # txtfusion projector
        mm = m[:, :CTX.txtlen]
        return mm[..., None] if mm.shape[0] in (1, x.shape[0]) else None
    return None


def _kron_apply(x, w1, w2):
    """x @ kron(w1, w2).T without materializing the Kronecker product.

    w1: (a1, b1), w2: (a2, b2); x last dim must be b1*b2; result a1*a2.
    Identity: for X = x reshaped (…, b1, b2), Y = w1 @ X @ w2.T reshaped back —
    the same math ComfyUI's own LoKr adapter computes on weights, applied to
    activations.
    """
    a1, b1 = w1.shape
    a2, b2 = w2.shape
    lead = x.shape[:-1]
    xr = x.reshape(*lead, b1, b2)
    t = torch.einsum("...ij,pj->...ip", xr, w2)   # (…, b1, a2)
    yr = torch.einsum("...ip,qi->...qp", t, w1)   # (…, a1, a2)
    return yr.reshape(*lead, a1 * a2)


class _RegionalLoRAMixin:
    """Mixin prepended onto the layer's own class; adds masked LoRA deltas
    (standard low-rank and LoKr / Kronecker adapters)."""

    def forward(self, x, *args, **kwargs):
        y = super().forward(x, *args, **kwargs)
        adapters = getattr(self, "regional_adapters", None)
        tm = CTX.token_masks
        if not adapters or not tm or not torch.is_tensor(x) \
                or x.ndim not in (3, 4):
            return y
        cast = self.__dict__.setdefault("_k2b_cast", {})
        flat_x = None
        owns_output = False
        for uid, adapter in adapters.items():
            m = tm.get(uid)
            if m is None:
                continue
            plan = _token_plan(uid, m, x, y)
            if plan is None or (plan.indices is not None and plan.indices.numel() == 0):
                continue
            kind, wa, wb, scale = adapter
            ck = (uid, x.device, x.dtype)
            if ck not in cast:
                cast[ck] = (wa.to(device=x.device, dtype=x.dtype),
                            wb.to(device=x.device, dtype=x.dtype))
            a, b = cast[ck]
            if plan.indices is not None:
                if flat_x is None:
                    flat_x = x.reshape(-1, x.shape[-1])
                adapter_x = flat_x.index_select(0, plan.indices)
            else:
                adapter_x = x
            if kind == "lokr":
                if a.shape[1] * b.shape[1] != x.shape[-1]:
                    continue  # factorization doesn't fit this layer
                delta = _kron_apply(adapter_x, a, b)
            else:
                delta = F.linear(F.linear(adapter_x, a), b)
            if plan.indices is None:
                y = y + delta * (scale * plan.weights)
                owns_output = True
            else:
                # Only selected rows receive this adapter. Clone once to avoid
                # mutating a base projection that may be aliased elsewhere.
                if not owns_output or not y.is_contiguous():
                    y = y.clone(memory_format=torch.contiguous_format)
                    owns_output = True
                y.view(-1, y.shape[-1]).index_add_(
                    0, plan.indices, delta * (scale * plan.weights))
        return y


class _TokenPlan:
    __slots__ = ("indices", "weights")

    def __init__(self, indices, weights):
        self.indices = indices
        self.weights = weights


def _token_plan(uid, mask, x, y):
    """Cache the exact nonzero support for each layer layout in this run.

    No threshold is applied to grey/feathered masks. Zero rows cannot contribute
    to a token-wise linear adapter, so their matrix multiplications can be
    omitted. Dense support uses the original path to avoid gather/scatter costs.
    """
    key = (uid, tuple(x.shape[:-1]), x.device, y.device, y.dtype)
    if key not in CTX.token_plans:
        mm = _adapt_mask(mask, x)
        plan = None
        if mm is not None:
            mm = mm.to(device=x.device)
            flat = mm.squeeze(-1).expand(x.shape[:-1]).reshape(-1)
            indices = torch.nonzero(flat != 0, as_tuple=True)[0]
            if indices.numel() > flat.numel() * 0.75 or x.device != y.device:
                plan = _TokenPlan(None, mm.to(device=y.device, dtype=y.dtype))
            else:
                weights = flat.index_select(0, indices).to(dtype=y.dtype)[:, None]
                plan = _TokenPlan(indices, weights)
        CTX.token_plans[key] = plan
    return CTX.token_plans[key]


_CLS_CACHE: dict[type, type] = {}


def _regional_class(base_cls: type) -> type:
    if base_cls not in _CLS_CACHE:
        _CLS_CACHE[base_cls] = type(
            "Regional" + base_cls.__name__, (_RegionalLoRAMixin, base_cls), {}
        )
    return _CLS_CACHE[base_cls]


def _patched_linear(base: torch.nn.Module) -> torch.nn.Module:
    """Shallow-copy `base` and swap in the regional subclass.

    Parameters/buffers dicts are shared with the original, so weight keys,
    ModelPatcher weight patching, and lowvram casting all behave exactly as
    before. Each ApplyRegional execution gets its own copy with its own
    `regional_adapters`, so model clones never interfere.
    """
    patched = copy.copy(base)
    if isinstance(base, _RegionalLoRAMixin):
        patched.regional_adapters = dict(getattr(base, "regional_adapters", {}))
    else:
        patched.__class__ = _regional_class(base.__class__)
        patched.regional_adapters = {}
    return patched


def _inject_lora(model_patcher, lora_sd: dict, uid: str, strength: float,
                 unmaskable: str = "skip") -> int:
    """Attach one LoRA (as adapter `uid`) onto the DiT block Linears.

    Keys that hit layers OUTSIDE the token sequence (tmlp/tproj timestep
    embedders, etc. — trained by newer ai-toolkit configs) can't be
    region-masked: the timestep vector modulates the whole image through
    AdaLN, so there is no per-token axis. `unmaskable`:
      "skip"           -> leave them out (default; logged as INFO)
      "apply globally" -> merge them as normal weight patches via ComfyUI's
                          own LoRA loader (affects the WHOLE image)
    """
    groups = _normalize_lora_sd(lora_sd)

    diffusion_model = model_patcher.get_model_object("diffusion_model")
    lookup: dict[str, str] = {}
    full_lookup: dict[str, str] = {}
    for name, mod in diffusion_model.named_modules():
        if (hasattr(mod, "weight") and callable(mod)
                and getattr(mod, "weight", None) is not None
                and mod.weight.ndim == 2):
            full_lookup[name] = name
            full_lookup[name.replace(".", "_")] = name
            if name.startswith(("blocks.", "txtfusion.", "txtmlp", "first",
                                "last")):
                lookup[name] = name
                lookup[name.replace(".", "_")] = name

    patched, skipped = 0, []
    global_keys: dict[str, str] = {}
    for key, g in groups.items():
        path = lookup.get(key) or lookup.get(key.replace(".", "_"))
        if path is None:
            gpath = (full_lookup.get(key)
                     or full_lookup.get(key.replace(".", "_")))
            if gpath is not None:
                # real layer, just not a token-sequence one
                global_keys[g["orig"]] = "diffusion_model." + gpath + ".weight"
            else:
                skipped.append(key)
            continue
        full = "diffusion_model." + path
        target = _patched_linear(model_patcher.get_model_object(full))
        if g["type"] == "lokr":
            target.regional_adapters[uid] = (
                "lokr", g["w1"], g["w2"],
                float(strength) * float(g["alpha_scale"]),
            )
        else:
            rank = g["down"].shape[0]
            alpha = g.get("alpha", rank)
            target.regional_adapters[uid] = (
                "lora", g["down"], g["up"],
                float(strength) * float(alpha) / float(rank),
            )
        model_patcher.add_object_patch(full, target)
        patched += 1

    if global_keys:
        if unmaskable == "apply globally":
            import comfy.lora
            patches = comfy.lora.load_lora(lora_sd, global_keys,
                                           log_missing=False)
            if patches:
                model_patcher.add_patches(patches, float(strength))
            logging.info(
                "[Krea2TwoCharacter] lora '%s': %d timestep/embedding layers "
                "applied GLOBALLY (whole image) — they can't be "
                "region-masked.", uid, len(global_keys))
        else:
            logging.info(
                "[Krea2TwoCharacter] lora '%s': %d timestep/embedding layers "
                "skipped (e.g. %s) — they modulate the whole image and "
                "can't be region-masked. Set unmaskable_layers to 'apply "
                "globally' on Apply Regional to merge them.",
                uid, len(global_keys), list(global_keys)[:3])
    if skipped:
        logging.warning(
            "[Krea2TwoCharacter] lora '%s': %d keys didn't match any layer "
            "(e.g. %s) — is this a Krea 2 LoRA?", uid, len(skipped),
            skipped[:3]
        )
    return patched


def _latent_mask(mask: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
    """(B?, H, W) MASK in [0,1] -> flattened soft mask on the patch grid."""
    if mask.ndim == 2:
        mask = mask[None]
    m = mask[:1].float()[None]  # (1, 1, H, W)
    m = F.interpolate(m, size=(h, w), mode="area")[0, 0]
    return m.clamp(0, 1).reshape(-1).to(device)


def _exclusive_soft(soft):
    """Winner-take-all where region masks overlap on the token grid: each
    image token keeps only its strongest region. Stops two grown/feathered
    boxes from double-applying LoRAs (and sharing attention) in the gap
    between them.

    Depth priority: where strengths are (near-)equal — the common case of
    two hard masks overlapping at 1.0 — the EARLIER region in the list wins
    (top of the Builder's region list = front). Ties are resolved by EQUALITY (first region at the exact
    maximum wins); any genuine strength difference, however small, wins
    outright — no epsilon bias distorting real comparisons."""
    if soft.shape[0] < 2:
        return soft
    mx = soft.max(dim=0, keepdim=True).values
    winner = torch.zeros_like(soft, dtype=torch.bool)
    winner.scatter_(0, (soft == mx).float().argmax(dim=0, keepdim=True),
                    True)
    return soft * winner


def _build_allow(segments, region_img_hard, imglen, restrict_img_attn, device,
                 release_text=False):
    """(L, L) bool, True = may attend. segments: list of (start, end) text spans;
    segments[0] is the base prompt. region_img_hard: (R, imglen) bool.
    release_text: every image token may read every region prompt (the late
    phase of "Soft"); prompts stay isolated from each other."""
    txtlen = segments[-1][1]
    L = txtlen + imglen
    i0, i1 = txtlen, txtlen + imglen
    allow = torch.zeros(L, L, dtype=torch.bool, device=device)

    for s, e in segments:  # prompts are mutually isolated
        allow[s:e, s:e] = True

    b0, b1 = segments[0]  # base prompt <-> every image token
    allow[b0:b1, i0:i1] = True
    allow[i0:i1, b0:b1] = True

    for r, (s, e) in enumerate(segments[1:]):
        if release_text:
            allow[s:e, i0:i1] = True
            allow[i0:i1, s:e] = True
        else:
            allow[s:e, i0:i1] |= region_img_hard[r][None, :]
            allow[i0:i1, s:e] |= region_img_hard[r][:, None]

    if restrict_img_attn and region_img_hard.shape[0] > 0:
        shared = (region_img_hard.T.float() @ region_img_hard.float()) > 0
        bg = ~region_img_hard.any(dim=0)
        img_allow = shared | bg[None, :] | bg[:, None]
        img_allow |= torch.eye(imglen, dtype=torch.bool, device=device)
        allow[i0:i1, i0:i1] = img_allow
    else:
        allow[i0:i1, i0:i1] = True

    txt_allow = torch.zeros(txtlen, txtlen, dtype=torch.bool, device=device)
    for s, e in segments:
        txt_allow[s:e, s:e] = True
    return allow, txt_allow


def _feather_tokens(soft, h, w, sigma):
    """Gaussian-blur (R, h*w) token masks; edges replicate so a region that
    touches the image border doesn't fade out there. Complementary masks stay
    complementary, so two LoRAs cross-fade at their shared boundary."""
    if sigma <= 0 or soft.shape[0] == 0:
        return soft
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=soft.device)
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    kernel = (kernel / kernel.sum()).to(soft.dtype)
    m = soft.view(-1, 1, h, w)
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return m.view(soft.shape[0], -1).clamp(0, 1)


def _build_token_masks(bundle, segments, soft_masks, imglen, batch_rows, device):
    """Per-adapter (B, L, 1) masks. batch_rows: list of bool, True = regional
    (cond) rows, False = rows that must run the clean base model."""
    txtlen = segments[-1][1]
    L = txtlen + imglen
    row = {}

    def bump(uid, vec):
        row[uid] = torch.maximum(row[uid], vec) if uid in row else vec

    for spec in bundle["loras"]:
        uid, weight, seg_idx, mask_idx = (
            spec["uid"], spec["weight"], spec["seg"], spec["mask_idx"],
        )
        vec = torch.zeros(L, device=device)
        if mask_idx is None:  # global (base) lora: whole prompt + whole image
            vec[:txtlen] = weight
            if (bundle.get("base_loras_exclude_regions")
                    and soft_masks.shape[0] > 0):
                # style/base loras skin everything EXCEPT the subject regions
                union = soft_masks.max(dim=0).values
                vec[txtlen:] = weight * (1.0 - union)
            else:
                vec[txtlen:] = weight
        else:
            vec[txtlen:] = soft_masks[mask_idx] * weight
            if seg_idx is not None:
                s, e = segments[seg_idx]
                vec[s:e] = weight  # trigger words go through the LoRA too
        bump(uid, vec)

    out = {}
    if all(batch_rows):
        for uid, vec in row.items():
            out[uid] = vec[None, :, None]  # (1, L, 1) -> broadcasts to any batch
    else:
        rows = torch.tensor(batch_rows, device=device, dtype=torch.float32)
        for uid, vec in row.items():
            out[uid] = (rows[:, None] * vec[None, :])[:, :, None]  # (B, L, 1)
    return out



# ---------------------------------------------------------------------------
# The diffusion-model wrapper (WrappersMP.DIFFUSION_MODEL)
# ---------------------------------------------------------------------------


def _find_reference_latents(rest, kwargs):
    ref = kwargs.get("ref_latents")
    if ref is None:
        for a in rest:
            if isinstance(a, (list, tuple)) and len(a) > 0 and torch.is_tensor(a[0]):
                return a
    return ref


def _diffusion_wrapper(executor, x, timesteps, context, *args, **kwargs):
    # ComfyUI has inserted positional parameters into Krea 2's forward over
    # time. The options dict is either the `transformer_options` kwarg or the
    # last positional dict; everything else passes through untouched.
    transformer_options = kwargs.pop("transformer_options", None)
    rest = list(args)
    if transformer_options is None and rest and isinstance(rest[-1], dict):
        transformer_options = rest.pop()
    if transformer_options is None:
        transformer_options = {}
    bundle = transformer_options.get(BUNDLE_KEY)
    if bundle is not None and not transformer_options.get(CONSUMED_KEY):
        # Reference latents extend the image sequence; the region masks would
        # silently misalign, so fail loudly instead.
        ref = _find_reference_latents(rest, kwargs)
        if ref is not None and len(ref) > 0:
            raise RuntimeError(
                f"{LOG} Krea 2 reference latents are not supported together "
                "with the two-character composer.")
    if (
        bundle is None
        or context is None
        or transformer_options.get(CONSUMED_KEY)
        or context.shape[1] != bundle["txt_total"]
    ):
        # negative prompt / unrelated conditioning / chained duplicate
        if (bundle is not None and context is not None
                and not transformer_options.get(CONSUMED_KEY)
                and bundle.get("expects_negpip")
                and context.shape[1] == bundle["txt_total_with_sidecars"]
                and 0 in transformer_options.get("cond_or_uncond", [0])):
            # Positive conditioning still carries its NegPiP sidecars: the
            # NegPiP wrapper is missing, so the regions would silently do
            # nothing. Say so instead. (A zeroed negative has the same length
            # but runs as an uncond-only call, which is skipped above.)
            raise RuntimeError(
                f"{LOG} NegPiP sidecar tokens reached the regional pass. The "
                "NegPiP model patch is missing; connect the composer's model "
                "output directly to the sampler.")
        return executor(x, timesteps, context, *rest,
                        transformer_options=transformer_options, **kwargs)

    model = executor.class_obj  # the SingleStreamDiT instance
    patch = model.patch
    # Mirror _forward: a 5D latent (B, C, T, H, W) is flattened to
    # (B*T, C, H, W) before patchify, so the DiT's effective batch is B*T.
    bs = x.shape[0] * x.shape[2] if x.ndim == 5 else x.shape[0]
    h_ = -(-x.shape[-2] // patch)  # matches pad_to_patch_size + patchify
    w_ = -(-x.shape[-1] // patch)
    imglen = h_ * w_
    device = x.device

    # which batch rows are regional (cond) vs must stay clean (uncond)?
    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    chunks = max(len(cond_or_uncond), 1)
    per = max(bs // chunks, 1)
    batch_rows = []
    for c in cond_or_uncond:
        batch_rows.extend([c == 0] * per)
    batch_rows = (batch_rows + [True] * bs)[:bs]

    segments = bundle["segments"]
    txtlen = bundle["txt_total"]
    L = txtlen + imglen

    # Run-level cache: masks depend only on geometry + batch layout, so build
    # them once instead of every step. It lives in a _RunCache object because
    # ComfyUI copies every nested dict of transformer_options per model call;
    # a key written into the bundle dict itself would be thrown away.
    ck = (h_, w_, bs, tuple(batch_rows), str(device), txtlen)
    cache = bundle["cache"]
    rt = cache.rt
    if rt is None or rt.get("key") != ck:
        rt = {"key": ck, "additive": {}, "token_masks": None, "token_plans": {},
              "masks_built": False}
        cache.rt = rt

    # Scheduled restrict: hard image<->image isolation while identity forms,
    # then release so the late steps can integrate seams and lighting.
    # Balanced has no schedule. Avoid a GPU-to-CPU synchronization every step
    # just to read a timestep that neither branch will use.
    scheduled = (
        bool(bundle["restrict_img_attn"]) and bundle.get("restrict_end_sigma", -1.0) >= 0
    ) or bundle.get("release_text_sigma", -1.0) >= 0
    t0 = float(timesteps.reshape(-1)[0]) if scheduled else 0.0
    restrict_now = bool(bundle["restrict_img_attn"]) and (
        t0 >= bundle.get("restrict_end_sigma", -1.0) - 1e-9)
    # "Soft": region prompts are isolated while the layout forms, then every
    # image token may read both prompts. LoRA routing stays regional.
    release_sigma = bundle.get("release_text_sigma", -1.0)
    release_now = release_sigma >= 0 and t0 < release_sigma - 1e-9
    phase = (restrict_now, release_now)
    if rt.get("phase") != phase:
        # only the attention masks change: token masks stay
        rt["phase"] = phase
        rt["masks_built"] = False
        rt["additive"] = {}

    if not rt["masks_built"]:
        soft = (
            torch.stack([_latent_mask(m, h_, w_, device) for m in bundle["masks"]])
            if bundle["masks"] else torch.zeros(0, imglen, device=device)
        )
        if bundle.get("exclusive_masks", True):
            soft = _exclusive_soft(soft)
        hard = soft > 0.5

        allow, txt_allow = _build_allow(
            segments, hard, imglen, restrict_now, device, release_text=release_now
        )
        # When every row is regional (Turbo at CFG 1), keep batch dim 1 so
        # the mask broadcasts against any caller batch. Only expand per row
        # when cond/uncond rows are actually mixed.
        if all(batch_rows):
            joint = allow[None, None]      # (1, 1, L, L)
            txtm = txt_allow[None, None]
        else:
            joint = allow[None].repeat(bs, 1, 1)
            txtm = txt_allow[None].repeat(bs, 1, 1)
            for i, regional in enumerate(batch_rows):
                if not regional:
                    joint[i] = True
                    txtm[i] = True
            joint = joint[:, None]  # (B, 1, L, L)
            txtm = txtm[:, None]
        rt.update(soft=soft, joint=joint, txtm=txtm, masks_built=True)
    soft = rt["soft"]
    joint = rt["joint"]
    txtm = rt["txtm"]
    additive_cache = rt["additive"]
    base_bias = float(bundle.get("base_bias", 0.0))
    b0, b1 = segments[0]

    def _additive(kind, sel: torch.Tensor, dtype):
        key = (kind, dtype)
        if key not in additive_cache:
            neg = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
            add = torch.zeros(sel.shape, dtype=dtype, device=sel.device).masked_fill_(~sel, neg)
            if kind == "joint" and base_bias != 0.0:
                # Base strength: scale how much the characters' image tokens
                # attend to the shared prompt (softmax(s + log w) weights those
                # keys by w). Background tokens and uncond rows are untouched.
                img_rows = torch.nonzero((soft > 0.5).any(dim=0)).flatten() + txtlen
                # A size-1 mask broadcasts when every row is conditional,
                # but can also describe an unconditional-only batch of one.
                # Decide from the row roles, not from the mask's shape.
                batch = [0] if all(batch_rows) else [
                    i for i, regional in enumerate(batch_rows) if regional]
                for i in batch:
                    add[i, :, img_rows, b0:b1] += base_bias
            additive_cache[key] = add
        return additive_cache[key]

    prev_override = transformer_options.get("optimized_attention_override")

    def override(func, q, k, v, heads, mask=None, **kw):
        seq = k.shape[2] if k.ndim == 4 else k.shape[1]
        qb = q.shape[0]
        sel = kind = None
        if seq == L:
            sel, kind = joint, "joint"
        elif seq == txtlen and txtlen != model.txtlayers:
            sel, kind = txtm, "txt"  # txtfusion refiner blocks
        # A sub-call with a batch the per-row mask can't broadcast to runs
        # unmasked rather than crashing (stock behaviour for that call).
        if sel is not None and sel.shape[0] not in (1, qb):
            sel = None
        if sel is not None and mask is None:
            mask = _additive(kind, sel, q.dtype)
        if prev_override is not None:
            return prev_override(func, q, k, v, heads, mask=mask, **kw)
        return func(q, k, v, heads, mask=mask, **kw)

    if rt["token_masks"] is None:
        lora_soft = soft
        feather = float(bundle.get("lora_feather_px", 0) or 0)
        if feather > 0 and bundle["masks"]:
            px_per_token = bundle["masks"][0].shape[-2] / h_
            lora_soft = _feather_tokens(soft, h_, w_, feather / px_per_token / 2.0)
        rt["token_masks"] = _build_token_masks(
            bundle, segments, lora_soft, imglen, batch_rows, device
        )

    transformer_options[CONSUMED_KEY] = True
    transformer_options["optimized_attention_override"] = override
    CTX.token_masks = rt["token_masks"]
    CTX.token_plans = rt["token_plans"]
    CTX.txtlen = txtlen
    CTX.imglen = imglen
    CTX.txtlayers = getattr(model, "txtlayers", -1)
    try:
        return executor(x, timesteps, context, *rest,
                        transformer_options=transformer_options, **kwargs)
    finally:
        CTX.clear()
        transformer_options.pop(CONSUMED_KEY, None)
        if prev_override is None:
            transformer_options.pop("optimized_attention_override", None)
        else:
            transformer_options["optimized_attention_override"] = prev_override


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layout_chunks(chunks):
    """Concatenate prompt chunks and compute segments on the stripped layout.

    chunks: list of (cond_tensor (1, seq, dim), sidecar_tokens). Each chunk's
    sidecar tokens (NegPiP) sit at its END, which is where the NegPiP wrapper
    expects them when it reconstructs per-segment token positions.

    Returns (combined, segments, txt_total, txt_total_with_sidecars).
    `segments` index the sequence AFTER all sidecars have been stripped.
    """
    tensors, segments, off, full = [], [], 0, 0
    for cond, sidecar in chunks:
        sidecar = int(sidecar or 0)
        length = int(cond.shape[1])
        if not 0 <= sidecar < length:
            raise ValueError(f"{LOG} invalid sidecar count {sidecar} for a "
                             f"{length}-token prompt.")
        tensors.append(cond)
        segments.append((off, off + length - sidecar))
        off += length - sidecar
        full += length
    combined = torch.cat([t.to(tensors[0]) for t in tensors], dim=1)
    return combined, segments, off, full


def apply_regional(model, base, regions, *, restrict_img_attn,
                   restrict_end_percent=1.0, exclusive_masks=True,
                   unmaskable_layers="skip", base_loras=(),
                   base_loras_exclude_regions=False, release_text_percent=None,
                   base_strength=1.0, lora_feather_px=0):
    """Patch a Krea 2 model for single-pass regional generation.

    base:    {"cond": (1, seq, dim), "extra": dict, "sidecar": int}
    regions: [{"cond": ..., "sidecar": int, "mask": (H, W), "loras": [entry]}]
             entry = {"sd": state_dict, "strength": float, "label": str}
    release_text_percent: release the region prompt isolation at this point
             of the schedule (None = never).
    base_strength: relative attention of regional image tokens to the shared
             prompt (1.0 = unchanged).
    lora_feather_px: soften the LoRA masks (not the attention masks) by about
             this many pixels.
    Returns (patched_model, conditioning).
    """
    base_strength = float(base_strength)
    if not math.isfinite(base_strength) or base_strength <= 0:
        raise ValueError(f"{LOG} base strength must be above 0.")
    m = model.clone()

    combined, segments, txt_total, txt_full = layout_chunks(
        [(base["cond"], base.get("sidecar", 0))]
        + [(r["cond"], r.get("sidecar", 0)) for r in regions])

    lora_specs = []
    for entry in base_loras or ():
        uid = f"k2tc{next(_UID)}"
        n = _inject_lora(m, entry["sd"], uid, entry["strength"])
        logging.info("%s base lora '%s': %d layers", LOG, entry["label"], n)
        lora_specs.append({"uid": uid, "weight": 1.0, "seg": None,
                           "mask_idx": None})
    for ridx, region in enumerate(regions):
        for entry in region.get("loras") or ():
            uid = f"k2tc{next(_UID)}"
            n = _inject_lora(m, entry["sd"], uid, entry["strength"],
                             unmaskable=unmaskable_layers)
            logging.info("%s region %d lora '%s': %d layers", LOG, ridx,
                         entry["label"], n)
            if n == 0:
                logging.warning("%s lora '%s' patched no layers; is it a "
                                "Krea 2 LoRA?", LOG, entry["label"])
            lora_specs.append({"uid": uid, "weight": 1.0, "seg": ridx + 1,
                               "mask_idx": ridx})

    # Convert the release point to sigma space so it tracks the real noise
    # schedule at any step count. -1 means restricted for the whole run.
    restrict_end_sigma = -1.0
    if restrict_img_attn and restrict_end_percent < 0.999:
        try:
            ms = model.model.model_sampling
            restrict_end_sigma = float(ms.percent_to_sigma(restrict_end_percent))
        except Exception:
            restrict_end_sigma = -1.0
    release_text_sigma = -1.0
    if release_text_percent is not None:
        try:
            ms = model.model.model_sampling
            release_text_sigma = float(ms.percent_to_sigma(release_text_percent))
        except Exception:
            release_text_sigma = -1.0  # unknown sampling: never release

    bundle = {
        "segments": segments,
        "txt_total": txt_total,
        "txt_total_with_sidecars": txt_full,
        "expects_negpip": txt_full != txt_total,
        "masks": [r["mask"] for r in regions],
        "restrict_img_attn": bool(restrict_img_attn),
        "restrict_end_sigma": restrict_end_sigma,
        "exclusive_masks": bool(exclusive_masks),
        "base_loras_exclude_regions": bool(base_loras_exclude_regions),
        "loras": lora_specs,
        "release_text_sigma": release_text_sigma,
        "base_bias": 0.0 if base_strength == 1.0 else math.log(base_strength),
        "lora_feather_px": max(0.0, float(lora_feather_px or 0)),
        "cache": _RunCache(),
    }
    to = m.model_options.setdefault("transformer_options", {})
    to[BUNDLE_KEY] = bundle
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, BUNDLE_KEY,
                           _diffusion_wrapper)
    return m, [[combined, dict(base.get("extra") or {})]]
