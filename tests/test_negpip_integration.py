"""NegPiP + regional engine, against the REAL NegPiP code.

Skipped unless NEGPIP_PATH points at a comfyui-negpip-zimage checkout:

    NEGPIP_PATH=/path/to/comfyui-negpip-zimage python3 -m unittest tests.test_negpip_integration -v

Runs NegPiP's own model patch, sidecar format and diffusion wrapper with the
composer's regional wrapper inside it, the way ComfyUI stacks them.
"""

import importlib
import types
import unittest

import torch

from tests import helpers

helpers.load_package()
regional = importlib.import_module(f"{helpers.PACKAGE}.engine.regional")
negpip = helpers.load_negpip()

IMGLEN = 6 * 8


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.wv = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(self.attn.wv.weight)

    def forward(self, x, transformer_options={}):
        return self.attn.wv(x)


def fake_krea2_clip():
    model = types.SimpleNamespace(qwen3vl_4b=object(), encode_token_weights=lambda *a, **k: None)
    clip = types.SimpleNamespace(cond_stage_model=model, tokenizer=types.SimpleNamespace())
    clip.clone = lambda: types.SimpleNamespace(cond_stage_model=model, tokenizer=clip.tokenizer)
    return clip


def cond(n, seed):
    return torch.randn(1, n, helpers.FLAT, generator=torch.Generator().manual_seed(seed)) * 0.02


@unittest.skipIf(negpip is None, "set NEGPIP_PATH to run against the real NegPiP code")
class NegPipIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.dm = helpers.FakeDM()
        self.dm.blocks.append(Block())
        np_model, _ = negpip.patch_krea2_negpip(helpers.FakePatcher(self.dm), fake_krea2_clip())
        left = torch.zeros(96, 128)
        left[:, :64] = 1
        base = negpip._make_sidecar(cond(10, 1), [3, 6])     # scene with two negatives
        char1 = negpip._make_sidecar(cond(4, 2), [1])        # inline (beard:-x)
        char2 = cond(6, 3)                                   # no negatives
        self.patched, self.positive = regional.apply_regional(
            np_model,
            {"cond": base, "extra": {}, "sidecar": base.shape[1] - 10},
            [{"cond": char1, "sidecar": char1.shape[1] - 4, "mask": left, "loras": []},
             {"cond": char2, "sidecar": 0, "mask": 1 - left, "loras": []}],
            restrict_img_attn=False)

    def test_negpip_is_outermost_and_regions_see_the_stripped_sequence(self):
        to = self.patched.transformer_options_for_sampling()
        pe = __import__("comfy.patcher_extension", fromlist=["x"])
        stack = pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, to)
        self.assertIs(stack[0], negpip.krea2_negpip_wrapper)
        self.assertIs(stack[-1], regional._diffusion_wrapper)

        seen = {}

        def inside(context, to):
            seen["len"] = context.shape[1]
            seen["positions"] = to["krea2_negpip"]["_negative_positions"]
            seen["override"] = "optimized_attention_override" in to
            x = torch.ones(1, 20 + IMGLEN, 4)
            seen["v"] = self.dm.blocks[0](x, transformer_options=to)[0, :, 0]

        self.dm.inside = inside
        self.assertEqual(self.positive[0][0].shape[1], 22)  # 20 tokens + 2 sidecars
        self.dm.run(self.patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), self.positive[0][0])
        self.assertEqual(seen["len"], 20)
        self.assertTrue(seen["override"], "regional attention mask was not installed")
        # base negatives at 3 and 6; character 1 starts at 10, so its negative is 11
        self.assertEqual(seen["positions"], [[3, 6, 11]])
        flipped = (seen["v"] < 0).nonzero().flatten().tolist()
        self.assertEqual(flipped, [3, 6, 11])

    def test_mixed_cond_uncond_batch(self):
        seen = {}

        def inside(context, to):
            seen["len"] = context.shape[1]
            seen["positions"] = to["krea2_negpip"]["_negative_positions"]
            q = torch.zeros(2, 2, 20 + IMGLEN, 4)
            masks = {}
            to["optimized_attention_override"](
                lambda q, k, v, heads, mask=None, **kw: masks.setdefault("m", mask), q, q, q, 2)
            seen["mask"] = masks["m"]

        self.dm.inside = inside
        context = torch.cat([self.positive[0][0], torch.zeros_like(self.positive[0][0])])
        self.dm.run(self.patched, torch.zeros(2, 16, 12, 16), torch.tensor([1.0, 1.0]), context,
                    cond_or_uncond=(0, 1))
        self.assertEqual(seen["len"], 20)
        self.assertEqual(seen["positions"], [[3, 6, 11], []])
        self.assertTrue((seen["mask"][1] == 0).all())      # uncond row unmasked
        self.assertFalse((seen["mask"][0] == 0).all())     # cond row regional


if __name__ == "__main__":
    unittest.main()
