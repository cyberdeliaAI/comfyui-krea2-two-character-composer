"""Regional engine with real torch and the ComfyUI wrapper semantics."""

import sys
import unittest

import torch

from tests import helpers

helpers.load_package()
regional = sys.modules[f"{helpers.PACKAGE}.engine.regional"] if f"{helpers.PACKAGE}.engine.regional" in sys.modules else None
if regional is None:
    import importlib
    regional = importlib.import_module(f"{helpers.PACKAGE}.engine.regional")

H, W = 96, 128           # pixels -> 12x16 latent -> 6x8 tokens (patch 2)
IMGLEN = 6 * 8


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.wv = torch.nn.Linear(8, 8, bias=False)


def lora(rank=2, value=1.0):
    return {
        "diffusion_model.blocks.0.attn.wv.lora_A.weight": torch.full((rank, 8), value),
        "diffusion_model.blocks.0.attn.wv.lora_B.weight": torch.full((8, rank), value),
    }


def half_masks(percent=50):
    left = torch.zeros(H, W)
    left[:, : round(W * percent / 100)] = 1
    return left, 1 - left


def chunk(n, dim=8, sidecar=0):
    return torch.randn(1, n + sidecar, dim)


def build(sidecars=(0, 0, 0), loras=True, separation="Balanced", **options):
    dm = helpers.FakeDM()
    dm.blocks.append(Block())
    patcher = helpers.FakePatcher(dm)
    left, right = half_masks()
    entry = lambda name: {"sd": lora(), "strength": 1.0, "label": name}
    base = {"cond": chunk(5, sidecar=sidecars[0]), "extra": {"pooled_output": None}, "sidecar": sidecars[0]}
    regions = [
        {"cond": chunk(3, sidecar=sidecars[1]), "sidecar": sidecars[1], "mask": left,
         "loras": [entry("a")] if loras else []},
        {"cond": chunk(4, sidecar=sidecars[2]), "sidecar": sidecars[2], "mask": right,
         "loras": [entry("b")] if loras else []},
    ]
    patched, positive = regional.apply_regional(
        patcher, base, regions, restrict_img_attn=separation == "Strong", restrict_end_percent=0.5,
        **options)
    return dm, patcher, patched, positive


def capture_mask(transformer_options, L, batch=1):
    seen = {}

    def func(q, k, v, heads, mask=None, **kw):
        seen["mask"] = mask
        return q

    q = torch.zeros(batch, 2, L, 4)
    transformer_options["optimized_attention_override"](func, q, q, q, 2)
    return seen["mask"]


class LayoutTests(unittest.TestCase):
    def test_segments_are_laid_out_on_the_sequence_without_sidecars(self):
        combined, segments, total, full = regional.layout_chunks(
            [(chunk(5, sidecar=1), 1), (chunk(3), 0), (chunk(4, sidecar=2), 2)])
        self.assertEqual(combined.shape[1], 6 + 3 + 6)
        self.assertEqual(segments, [(0, 5), (5, 8), (8, 12)])
        self.assertEqual((total, full), (12, 15))

    def test_invalid_sidecar_counts_fail(self):
        for bad in (-1, 3):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                regional.layout_chunks([(chunk(3), bad)])


class WrapperTests(unittest.TestCase):
    def test_base_strength_only_changes_conditional_region_rows(self):
        for strength in (0.5, 2.0):
            for roles, batch in (((0,), 1), ((1,), 1), ((0,), 2), ((1,), 2), ((0, 1), 2), ((1, 0), 4)):
                with self.subTest(strength=strength, roles=roles, batch=batch):
                    dm, _, patched, positive = build(loras=False, base_strength=strength)
                    bundle = patched.model_options["transformer_options"][regional.BUNDLE_KEY]
                    first = torch.zeros(H, W); first[:, :32] = 1
                    second = torch.zeros(H, W); second[:, 96:] = 1
                    bundle["masks"] = [first, second]
                    captured = []
                    dm.inside = lambda ctx, to: captured.append(capture_mask(to, 12 + IMGLEN, batch))
                    context = positive[0][0].expand(batch, -1, -1).clone()
                    per_group = batch // len(roles)
                    for i, role in enumerate(roles):
                        if role == 1: context[i * per_group:(i + 1) * per_group] = 0
                    dm.run(patched, torch.zeros(batch, 16, 12, 16), torch.ones(batch), context,
                           cond_or_uncond=roles)
                    mask = captured[0]
                    columns = torch.arange(IMGLEN) % 8
                    own = (columns < 2) | (columns >= 6)
                    for row in range(batch):
                        role = roles[row // per_group]
                        actual = mask[0 if mask.shape[0] == 1 else row, 0]
                        if role == 1:
                            self.assertTrue((actual == 0).all(), "unconditional predictions must remain unbiased")
                        else:
                            expected = torch.zeros(IMGLEN, 5)
                            expected[own] = torch.log(torch.tensor(strength))
                            torch.testing.assert_close(actual[12:, :5], expected)

    def test_soft_release_opens_prompts_but_keeps_lora_routing(self):
        dm, _, patched, positive = build(release_text_percent=0.6)
        seen = []
        def inside(ctx, to):
            mask = capture_mask(to, 12 + IMGLEN)
            gates = {key: value.clone() for key, value in regional.CTX.token_masks.items()}
            seen.append((mask, gates))
        dm.inside = inside
        for sigma in (1.0, 0.2):
            dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([sigma]), positive[0][0])
        self.assertFalse((seen[0][0][0, 0, 12:, 5:12] == 0).all())
        self.assertTrue((seen[1][0][0, 0, 12:, 5:12] == 0).all())
        for uid in seen[0][1]:
            torch.testing.assert_close(seen[0][1][uid], seen[1][1][uid])

    def test_feathering_crossfades_without_losing_total_strength(self):
        first = torch.tensor([1., 1., 0., 0.])
        result = regional._feather_tokens(torch.stack([first, 1 - first]), 1, 4, 1.0)
        torch.testing.assert_close(result.sum(0), torch.ones(4))
        self.assertGreater(result[0, 2].item(), 0)
        self.assertLess(result[0, 1].item(), 1)

    def test_original_model_is_untouched_and_wrapper_is_installed(self):
        dm, patcher, patched, positive = build()
        self.assertFalse(patcher.wrappers)
        self.assertIn(regional.BUNDLE_KEY, patched.wrappers["diffusion_model"])
        self.assertEqual(positive[0][0].shape[1], 12)
        bundle = patched.model_options["transformer_options"][regional.BUNDLE_KEY]
        self.assertEqual(bundle["segments"], [(0, 5), (5, 8), (8, 12)])
        self.assertFalse(bundle["expects_negpip"])

    def test_attention_mask_and_lora_routing_during_a_call(self):
        dm, _, patched, positive = build()
        context = positive[0][0]
        L = 12 + IMGLEN
        seen = {}

        def inside(ctx, to):
            mask = capture_mask(to, L)[0, 0]
            allowed = mask == 0
            img = slice(12, L)
            left_cols = torch.arange(IMGLEN) % 8 < 4
            # base prompt <-> every image token
            self.assertTrue(allowed[img, 0:5].all())
            # left image tokens read prompt 1 only, right ones prompt 2 only
            self.assertTrue(allowed[img, 5:8][left_cols].all())
            self.assertFalse(allowed[img, 5:8][~left_cols].any())
            self.assertTrue(allowed[img, 8:12][~left_cols].all())
            self.assertFalse(allowed[img, 8:12][left_cols].any())
            # prompts are isolated from each other
            self.assertFalse(allowed[5:8, 8:12].any())
            # Balanced: image tokens see each other
            self.assertTrue(allowed[img, img].all())
            # LoRA delta only on its own prompt and image tokens
            layer = patched.object_patches["diffusion_model.blocks.0.attn.wv"]
            x = torch.ones(1, L, 8)
            delta = (layer(x) - dm.blocks[0].attn.wv(x)).abs().sum(-1)[0]
            touched = delta > 0
            self.assertTrue(touched[5:12].all())
            self.assertFalse(touched[0:5].any())
            self.assertTrue(touched[12:].all())  # two LoRAs cover all image tokens
            seen["ok"] = True

        dm.inside = inside
        dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), context)
        self.assertTrue(seen.get("ok"))
        self.assertEqual(regional.CTX.token_masks, {})  # cleared after the call

    def test_masks_are_built_once_across_model_calls(self):
        from unittest.mock import patch
        dm, _, patched, positive = build()
        with patch.object(regional, "_build_allow", wraps=regional._build_allow) as spy:
            for _ in range(3):
                dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), positive[0][0])
        self.assertEqual(spy.call_count, 1)

    def test_token_selections_survive_copied_options_and_attention_phase_changes(self):
        from unittest.mock import patch
        dm, _, patched, positive = build(separation="Strong")
        layer = patched.object_patches["diffusion_model.blocks.0.attn.wv"]
        dm.inside = lambda ctx, to: layer(torch.ones(1, 12 + IMGLEN, 8))
        with patch.object(regional.torch, "nonzero", wraps=torch.nonzero) as find:
            for sigma in (1.0, .2, .1):
                dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([sigma]), positive[0][0])
            self.assertEqual(find.call_count, 2, "one selection per adapter, not per step")
        self.assertEqual(regional.CTX.token_plans, {})

    def test_adapter_selections_rebuild_for_mixed_batch_and_geometry(self):
        dm, _, patched, positive = build()
        layer = patched.object_patches["diffusion_model.blocks.0.attn.wv"]
        for batch, h, w, roles in ((1, 12, 16, (0,)), (2, 12, 16, (0, 1)), (2, 16, 12, (1, 0))):
            with self.subTest(batch=batch, geometry=(h, w), roles=roles):
                length = 12 + (h // 2) * (w // 2)
                def inside(ctx, to):
                    x = torch.ones(batch, length, 8)
                    output = layer(x)
                    base = dm.blocks[0].attn.wv(x)
                    for row, role in enumerate(roles):
                        if role == 1:
                            self.assertTrue(torch.equal(output[row], base[row]))
                        else:
                            self.assertTrue((output[row, 12:] != base[row, 12:]).any())
                dm.inside = inside
                dm.run(patched, torch.zeros(batch, 16, h, w), torch.ones(batch),
                       positive[0][0].expand(batch, -1, -1), cond_or_uncond=roles)

    def test_balanced_does_not_read_timestep_back_to_cpu(self):
        from unittest.mock import patch
        dm, _, patched, positive = build()
        with patch.object(torch.Tensor, "__float__", side_effect=AssertionError("unnecessary scalar synchronization")):
            dm.run(patched, torch.zeros(1, 16, 12, 16), torch.ones(1), positive[0][0])

    def test_strong_restricts_image_attention_until_release(self):
        dm, _, patched, positive = build(separation="Strong")
        L = 12 + IMGLEN
        results = []
        dm.inside = lambda ctx, to: results.append(capture_mask(to, L)[0, 0])
        for t in (1.0, 0.2):  # sigma 1.0 restricted, 0.2 after release at 0.5
            dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([t]), positive[0][0])
        left_cols = torch.arange(IMGLEN) % 8 < 4
        early = results[0][12:, 12:] == 0
        self.assertFalse(early[left_cols][:, ~left_cols].any())
        self.assertTrue((results[1][12:, 12:] == 0).all())

    def test_uncond_rows_run_clean(self):
        dm, _, patched, positive = build()
        L = 12 + IMGLEN
        dm.inside = lambda ctx, to: self.assertTrue((capture_mask(to, L, batch=2)[1] == 0).all())
        context = torch.cat([positive[0][0], torch.zeros_like(positive[0][0])])
        dm.run(patched, torch.zeros(2, 16, 12, 16), torch.tensor([1.0, 1.0]), context,
               cond_or_uncond=(0, 1))

    def test_other_lengths_pass_through_untouched(self):
        dm, _, patched, _ = build()
        dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), torch.zeros(1, 7, 8))
        self.assertNotIn("optimized_attention_override", dm.calls[-1][1])

    def test_sidecars_reaching_the_regional_pass_raise_for_cond_rows_only(self):
        dm, _, patched, positive = build(sidecars=(1, 0, 1))
        context = positive[0][0]  # 14 tokens incl. 2 sidecars, nobody strips them
        with self.assertRaisesRegex(RuntimeError, "NegPiP model patch is missing"):
            dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), context)
        # a zeroed negative of the same length, run on its own, is fine
        dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]),
               torch.zeros_like(context), cond_or_uncond=(1,))

    def test_reference_latents_are_rejected(self):
        dm, patched_src, patched, positive = build(loras=False)
        to = patched.transformer_options_for_sampling()
        pe = sys.modules["comfy.patcher_extension"]
        wrappers = pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, to)
        with self.assertRaisesRegex(RuntimeError, "reference latents"):
            pe.WrapperExecutor.new_class_executor(dm._forward, dm, wrappers).execute(
                torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), positive[0][0], None,
                [torch.zeros(1, 16, 12, 16)], to)


if __name__ == "__main__":
    unittest.main()
