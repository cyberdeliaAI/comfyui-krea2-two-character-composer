"""Test doubles for ComfyUI. Tests need real torch, nothing else.

Set COMFYUI_PATH to a ComfyUI checkout to use its real
comfy/patcher_extension.py (wrapper ordering) instead of the small stand-in,
and NEGPIP_PATH to a comfyui-negpip-zimage checkout to run the NegPiP
integration tests against the real NegPiP code.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "krea2_two_character_composer_under_test"
LORAS = ["characters/a.safetensors", "b.safetensors"]
TXT_LAYERS, TXT_DIM = 12, 2560  # real Krea 2 text tap layout
FLAT = TXT_LAYERS * TXT_DIM
logging.getLogger().setLevel(logging.ERROR)


def _patcher_extension():
    path = os.environ.get("COMFYUI_PATH")
    if path:
        spec = importlib.util.spec_from_file_location(
            "comfy.patcher_extension", Path(path) / "comfy" / "patcher_extension.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Same semantics as ComfyUI: wrappers are grouped by key in insertion
    # order and index 0 is the outermost call.
    module = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"
        CALC_COND_BATCH = "calc_cond_batch"

    def add_wrapper_with_key(kind, key, wrapper, options, is_model_options=False):
        if is_model_options:
            options = options.setdefault("transformer_options", {})
        options.setdefault("wrappers", {}).setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def get_all_wrappers(kind, options, is_model_options=False):
        if is_model_options:
            options = options.get("transformer_options", {})
        out = []
        for items in options.get("wrappers", {}).get(kind, {}).values():
            out.extend(items)
        return out

    def copy_nested(d):
        # dicts and lists are copied, every other object is shared
        out = dict(d)
        for key, value in d.items():
            if isinstance(value, dict):
                out[key] = copy_nested(value)
            elif isinstance(value, list):
                out[key] = list(value)
        return out

    def merge_nested_dicts(a, b, copy_dict1=True):
        merged = copy_nested(a) if copy_dict1 else a
        for key, value in b.items():
            if isinstance(value, dict):
                merged[key] = merge_nested_dicts(merged.setdefault(key, {}), value)
            elif isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            else:
                merged[key] = value
        return merged

    class WrapperExecutor:
        def __init__(self, original, class_obj, wrappers, idx):
            self.original, self.class_obj = original, class_obj
            self.wrappers, self.idx = list(wrappers), idx

        def __call__(self, *args, **kwargs):
            return WrapperExecutor(self.original, self.class_obj, self.wrappers,
                                   self.idx + 1).execute(*args, **kwargs)

        def execute(self, *args, **kwargs):
            if self.idx == len(self.wrappers):
                return self.original(*args, **kwargs)
            return self.wrappers[self.idx](self, *args, **kwargs)

        @classmethod
        def new_class_executor(cls, original, class_obj, wrappers, idx=0):
            return cls(original, class_obj, wrappers, idx)

    module.WrappersMP = WrappersMP
    module.add_wrapper_with_key = add_wrapper_with_key
    module.get_all_wrappers = get_all_wrappers
    module.merge_nested_dicts = merge_nested_dicts
    module.WrapperExecutor = WrapperExecutor
    return module


class Krea2:  # stands in for comfy.model_base.Krea2
    def __init__(self):
        self.latent_format = types.SimpleNamespace(latent_channels=16, spacial_downscale_ratio=8)
        self.model_sampling = types.SimpleNamespace(percent_to_sigma=lambda p: 1.0 - p)


def install_comfy_stubs():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    pe = _patcher_extension()
    model_base = types.ModuleType("comfy.model_base")
    model_base.Krea2 = Krea2
    utils = types.ModuleType("comfy.utils")
    utils.loaded = []

    def load_torch_file(path, safe_load=False):
        utils.loaded.append(path)
        return {"fixture": path}

    utils.load_torch_file = load_torch_file
    mm = types.ModuleType("comfy.model_management")
    mm.intermediate_device = lambda: torch.device("cpu")
    samplers = types.ModuleType("comfy.samplers")
    sd1 = types.ModuleType("comfy.sd1_clip")
    te = types.ModuleType("comfy.text_encoders")
    te.__path__ = []
    qwen = types.ModuleType("comfy.text_encoders.qwen_vl")
    te.qwen_vl = qwen
    for name, module in {"patcher_extension": pe, "model_base": model_base, "utils": utils,
                         "model_management": mm, "samplers": samplers, "sd1_clip": sd1,
                         "text_encoders": te}.items():
        setattr(comfy, name, module)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda kind: list(LORAS)
    folder_paths.get_full_path_or_raise = lambda kind, name: f"/loras/{name}"
    modules = {
        "comfy": comfy, "comfy.patcher_extension": pe, "comfy.model_base": model_base,
        "comfy.utils": utils, "comfy.model_management": mm, "comfy.samplers": samplers,
        "comfy.sd1_clip": sd1, "comfy.text_encoders": te, "comfy.text_encoders.qwen_vl": qwen,
        "folder_paths": folder_paths,
    }
    sys.modules.update(modules)
    return modules


def load_package():
    install_comfy_stubs()
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = package
    spec.loader.exec_module(package)
    return package


def load_negpip():
    """Load the real NegPiP pack the way ComfyUI does, or return None."""
    path = os.environ.get("NEGPIP_PATH")
    if not path:
        return None
    install_comfy_stubs()
    name = str(Path(path).resolve()).replace(".", "_x_")
    if name + ".src.negpip_krea2" in sys.modules:
        return sys.modules[name + ".src.negpip_krea2"]
    root = types.ModuleType(name)
    root.__path__ = [str(Path(path))]
    sys.modules[name] = root
    src = types.ModuleType(name + ".src")
    src.__path__ = [str(Path(path) / "src")]
    sys.modules[name + ".src"] = src
    for sub in ("prompt_merge", "negpip_krea2"):
        spec = importlib.util.spec_from_file_location(
            f"{name}.src.{sub}", Path(path) / "src" / f"{sub}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    # Register a stand-in for src.nodes (the real one needs the full ComfyUI
    # API) exposing exactly what the bridge looks for.
    nodes = types.ModuleType(name + ".src.nodes")
    nodes.NegPipPrompt = object
    nodes.merge_prompts = sys.modules[name + ".src.prompt_merge"].merge_prompts
    sys.modules[nodes.__name__] = nodes
    return sys.modules[name + ".src.negpip_krea2"]


class FakePatcher:
    """The parts of comfy.model_patcher.ModelPatcher the code uses."""

    def __init__(self, dm=None):
        self.model = Krea2()
        self.dm = dm or FakeDM()
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.object_patches = {}

    def clone(self):
        n = FakePatcher(self.dm)
        n.model = self.model
        n.model_options = copy.deepcopy(self.model_options)
        n.wrappers = {k: {k1: list(v1) for k1, v1 in v.items()} for k, v in self.wrappers.items()}
        n.object_patches = dict(self.object_patches)
        return n

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        parts = name.split(".")
        obj = self.dm if parts[0] == "diffusion_model" else getattr(self, parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.get(kind, {}).pop(key, None)

    def transformer_options_for_sampling(self, cond_or_uncond=(0,), uuids=None):
        """Mirror ComfyUI: sampler_helpers appends the patcher's own wrappers
        after the model_options ones, then calc_cond_batch merges
        transformer_options into a FRESH dict for every model call."""
        pe = sys.modules["comfy.patcher_extension"]
        options = copy.copy(self.model_options["transformer_options"])
        options["wrappers"] = pe.merge_nested_dicts(options.get("wrappers", {}), self.wrappers)
        to = pe.merge_nested_dicts({}, options, copy_dict1=False)
        to["cond_or_uncond"] = list(cond_or_uncond)
        to["uuids"] = list(uuids or [])
        return to


class FakeDM(torch.nn.Module):
    """Shape-only stand-in for Krea 2's SingleStreamDiT."""

    def __init__(self):
        super().__init__()
        self.patch = 2
        self.txtlayers = TXT_LAYERS
        self.txtdim = TXT_DIM
        self.txtfusion = torch.nn.Identity()
        self.txtmlp = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList()
        self.calls = []
        self.inside = None  # optional callback(context, transformer_options)

    def _unpack_context(self, context):
        return context

    def _forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                 transformer_options={}, **kwargs):
        self.calls.append((context.shape, dict(transformer_options)))
        if self.inside is not None:
            self.inside(context, transformer_options)
        return x

    def run(self, patcher, x, timesteps, context, **sampling):
        """Call _forward through the wrapper stack like Krea 2's forward does."""
        pe = sys.modules["comfy.patcher_extension"]
        to = patcher.transformer_options_for_sampling(**sampling)
        wrappers = pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, to)
        return pe.WrapperExecutor.new_class_executor(self._forward, self, wrappers).execute(
            x, timesteps, context, None, None, to)


class FakeClip:
    """Encodes text to a deterministic (1, tokens, FLAT) tensor."""

    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        n = max(1, len(tokens.split()))
        cond = torch.randn(1, n, FLAT, generator=torch.Generator().manual_seed(n)) * 0.01
        return [[cond, {"pooled_output": None}]]


def latent(height=96, width=128, batch=1):
    return {"samples": torch.zeros(batch, 16, height // 8, width // 8)}
