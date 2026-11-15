"""Regional adapter lifetime across cached outputs and sampling cleanup."""

import gc
import unittest
import weakref

import torch
import torch.nn.functional as F

from tests.test_engine import Block, IMGLEN, chunk, half_masks, helpers, lora, regional


class RunCacheTests(unittest.TestCase):
    def setUp(self):
        self.dm = helpers.FakeDM()
        self.dm.blocks.append(Block())
        self.patcher = helpers.FakePatcher(self.dm)
        self.layer = self.dm.blocks[0].attn.wv
        regional.CTX.clear()

    def tearDown(self):
        regional.CTX.clear()
        active = regional._ACTIVE_CACHE
        if active is not None and active() is not None:
            active().release()
        regional._ACTIVE_CACHE = None

    def compose(self, strength1=1.0, strength2=2.0):
        left, right = half_masks()
        return regional.apply_regional(
            self.patcher, {"cond": chunk(5), "extra": {}, "sidecar": 0},
            [{"cond": chunk(3), "sidecar": 0, "mask": left,
              "loras": [{"sd": lora(), "strength": strength1, "label": "a"}]},
             {"cond": chunk(4), "sidecar": 0, "mask": right,
              "loras": [{"sd": lora(), "strength": strength2, "label": "b"}]}],
            restrict_img_attn=False)

    @staticmethod
    def cache(patched):
        return patched.model_options["transformer_options"][regional.BUNDLE_KEY]["cache"]

    def run_model(self, patched, positive, device="cpu"):
        result = []
        x = torch.ones(1, 12 + IMGLEN, 8, device=device)
        self.dm.inside = lambda ctx, to: result.append(self.layer(x))
        self.dm.run(patched, torch.zeros(1, 16, 12, 16, device=device),
                    torch.tensor([1.0], device=device), positive[0][0].to(device))
        return result[0]

    def test_reexecutions_reuse_module_and_existing_pin_state(self):
        # Dynamic VRAM indexes state by module; preserve both identities.
        pins = self.layer._pins = {"weights": {"sentinel": object()}}
        caches = []
        for strength in (1.0, 0.5, 1.5):
            patched, _ = self.compose(strength, 2.0)
            self.assertNotIn("diffusion_model.blocks.0.attn.wv", patched.object_patches)
            self.assertIs(self.dm.blocks[0].attn.wv, self.layer)
            self.assertIs(self.layer._pins, pins)
            caches.append(self.cache(patched))
        ids = [set(c.adapters["diffusion_model.blocks.0.attn.wv"]) for c in caches]
        self.assertEqual(len(set().union(*ids)), 6)
        self.assertNotIn("regional_adapters", self.layer.__dict__)
        self.assertNotIn("_k2b_cast", self.layer.__dict__)

    def test_cleanup_rebuild_and_activation_release_previous_cache(self):
        first, pos1 = self.compose()
        cache1 = self.cache(first)
        self.run_model(first, pos1)
        self.assertTrue(cache1.cast)
        self.assertIsNotNone(cache1.rt)
        first.cleanup()
        self.assertEqual(cache1.cast, {})
        self.assertIsNone(cache1.rt)
        # Cleanup is idempotent, and cached outputs remain reusable.
        first.cleanup()
        self.run_model(first, pos1)
        self.assertTrue(cache1.cast)
        second, pos2 = self.compose(0.5, 1.5)
        self.run_model(second, pos2)
        self.assertEqual(cache1.cast, {})
        self.assertIsNone(cache1.rt)
        self.assertTrue(cache1.adapters)

    def test_alternating_cached_models_keep_character_strengths_and_regions(self):
        first, pos1 = self.compose(1.0, 2.0)
        second, pos2 = self.compose(3.0, 4.0)
        x = torch.ones(1, 12 + IMGLEN, 8)
        baseline = F.linear(x, self.layer.weight)
        a = self.run_model(first, pos1) - baseline
        first.cleanup()
        b = self.run_model(second, pos2) - baseline
        a2 = self.run_model(first, pos1) - baseline
        torch.testing.assert_close(a, a2)
        for output, strengths in ((a, (1.0, 2.0)), (b, (3.0, 4.0))):
            torch.testing.assert_close(output[:, :5], torch.zeros_like(output[:, :5]))
            torch.testing.assert_close(output[:, 5:8], torch.full_like(output[:, 5:8], 16 * strengths[0]))
            torch.testing.assert_close(output[:, 8:12], torch.full_like(output[:, 8:12], 16 * strengths[1]))
            image = output[:, 12:].reshape(1, 6, 8, 8)
            torch.testing.assert_close(image[:, :, :4], torch.full_like(image[:, :, :4], 16 * strengths[0]))
            torch.testing.assert_close(image[:, :, 4:], torch.full_like(image[:, :, 4:], 16 * strengths[1]))

    def test_shared_layer_runs_normally_outside_composer_wrapper(self):
        patched, positive = self.compose()
        x = torch.ones(1, 12 + IMGLEN, 8)
        baseline = F.linear(x, self.layer.weight)
        torch.testing.assert_close(self.layer(x), baseline)
        self.run_model(patched, positive)
        torch.testing.assert_close(self.layer(x), baseline)
        patched.cleanup()
        torch.testing.assert_close(self.layer(x), baseline)
        self.assertIsNone(regional.CTX.cast)
        self.assertEqual(regional.CTX.adapters, {})

    def test_forward_error_clears_context_and_allows_cleanup_and_reuse(self):
        patched, positive = self.compose()
        cache = self.cache(patched)

        def fail(ctx, to):
            self.layer(torch.ones(1, 12 + IMGLEN, 8))
            raise RuntimeError("test forward failure")

        self.dm.inside = fail
        with self.assertRaisesRegex(RuntimeError, "test forward failure"):
            self.dm.run(patched, torch.zeros(1, 16, 12, 16), torch.tensor([1.0]), positive[0][0])
        self.assertIsNone(regional.CTX.cast)
        self.assertEqual(regional.CTX.adapters, {})
        self.assertEqual(regional.CTX.token_masks, {})
        patched.cleanup()
        self.assertEqual(cache.cast, {})
        self.assertIsNone(cache.rt)
        self.run_model(patched, positive)
        self.assertTrue(cache.cast)

    def test_private_copy_path_does_not_inherit_bundle_or_cast_cache(self):
        self.compose()
        self.layer._k2b_cast = {"old": object()}
        copied = regional._patched_linear(self.layer)
        self.assertNotIn("_k2tc_path", copied.__dict__)
        self.assertNotIn("_k2b_cast", copied.__dict__)
        self.assertIs(copied.weight, self.layer.weight)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cleanup_releases_cuda_cast_tensor_references(self):
        self.dm.to("cuda")
        patched, positive = self.compose()
        cache = self.cache(patched)
        # Inference graphs must not retain the adapter tensors for backward.
        with torch.inference_mode():
            self.run_model(patched, positive, device="cuda")
        self.assertTrue(cache.cast)
        self.assertTrue(all(t.device.type == "cuda" for pair in cache.cast.values() for t in pair))
        references = [weakref.ref(t) for pair in cache.cast.values() for t in pair]
        # FakeDM records calls for other tests; real sampling doesn't keep them.
        self.dm.calls.clear()
        patched.cleanup()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertIsNone(cache.rt)


if __name__ == "__main__":
    unittest.main()
