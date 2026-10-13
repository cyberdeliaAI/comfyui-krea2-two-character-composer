"""Compare optimized adapters with the original dense, token-masked formula."""
import importlib
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests import helpers

helpers.load_package()
regional = importlib.import_module(f"{helpers.PACKAGE}.engine.regional")


class CheapProjection(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(*x.shape[:-1], 7, dtype=x.dtype, device=x.device)


class AliasProjection(torch.nn.Module):
    def forward(self, x):
        return x


class SparseLoRATests(unittest.TestCase):
    def setUp(self):
        regional.CTX.clear()
        regional.CTX.txtlen = 5
        regional.CTX.imglen = 7
        regional.CTX.txtlayers = 3
        torch.manual_seed(123)

    def tearDown(self):
        regional.CTX.clear()

    def adapter(self, kind, out=7, dtype=torch.float32):
        if kind == "lora":
            return (kind, torch.randn(3, 8, dtype=dtype), torch.randn(out, 3, dtype=dtype), .7)
        return (kind, torch.randn(1, 2, dtype=dtype), torch.randn(out, 4, dtype=dtype), .4)

    def masks(self, mixed=False):
        first = torch.tensor([0., 0., 1., 1., 0., 1., .5, 1e-6, 0., 0., 0., 0.])[None, :, None]
        second = torch.tensor([0., 0., 0., 0., 1., 0., .5, 0., 1., 1., 0., 0.])[None, :, None]
        if mixed:
            first = torch.cat([first, torch.zeros_like(first)])
            second = torch.cat([second, torch.zeros_like(second)])
        return {"one": first, "two": second}

    def reference(self, x, adapters, base):
        result = base(x)
        for uid, (kind, a, b, scale) in adapters.items():
            mask = regional._adapt_mask(regional.CTX.token_masks[uid], x)
            if mask is None:
                continue
            delta = regional._kron_apply(x, a, b) if kind == "lokr" else F.linear(F.linear(x, a), b)
            result = result + delta * (scale * mask.to(result.dtype))
        return result

    def test_all_layouts_match_dense_for_lora_and_lokr(self):
        # Joint, text-only, image-only, flattened text taps, 4D projector,
        # broadcast batches and mixed conditional/unconditional batches.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for kind in ("lora", "lokr"):
                for shape, mixed in (((1, 12, 8), False), ((2, 12, 8), False), ((2, 12, 8), True),
                                     ((1, 5, 8), False), ((2, 7, 8), True),
                                     ((5, 3, 8), False), ((10, 3, 8), False), ((10, 3, 8), True),
                                     ((2, 5, 3, 8), False), ((2, 5, 3, 8), True)):
                    with self.subTest(dtype=dtype, kind=kind, shape=shape, mixed=mixed):
                        regional.CTX.token_plans = {}
                        regional.CTX.token_masks = self.masks(mixed)
                        base = CheapProjection()
                        layer = regional._patched_linear(base)
                        layer.regional_adapters = {uid: self.adapter(kind, dtype=dtype) for uid in ("one", "two")}
                        x = torch.randn(shape, dtype=dtype)
                        actual = layer(x)
                        expected = self.reference(x, layer.regional_adapters, base)
                        torch.testing.assert_close(actual, expected, rtol=1e-5 if dtype == torch.float32 else .02,
                                                   atol=1e-5 if dtype == torch.float32 else .02)
                        for uid in layer.regional_adapters:
                            mask = regional._adapt_mask(regional.CTX.token_masks[uid], x)
                            if mask is not None:
                                self.assertIsNotNone(regional._token_plan(uid, regional.CTX.token_masks[uid], x, actual).indices)
                        if mixed and shape == (2, 12, 8):
                            self.assertFalse(actual[1].any(), "unconditional rows receive no adapter")

    def test_individual_adapters_never_touch_other_character_or_background(self):
        for uid in ("one", "two"):
            with self.subTest(uid=uid):
                regional.CTX.token_masks = self.masks()
                base = CheapProjection()
                layer = regional._patched_linear(base)
                layer.regional_adapters = {uid: self.adapter("lora")}
                x = torch.randn(1, 12, 8)
                output = layer(x)
                outside = regional.CTX.token_masks[uid][0, :, 0] == 0
                self.assertTrue(torch.equal(output[0, outside], base(x)[0, outside]))

    def test_selection_and_weights_are_cached_across_layers(self):
        regional.CTX.token_masks = self.masks()
        with patch.object(regional.torch, "nonzero", wraps=torch.nonzero) as find:
            for _ in range(3):
                layer = regional._patched_linear(CheapProjection())
                layer.regional_adapters = {"one": self.adapter("lora")}
                layer(torch.randn(2, 12, 8))
            self.assertEqual(find.call_count, 1)

    def test_projection_only_computes_selected_rows(self):
        regional.CTX.token_masks = self.masks()
        layer = regional._patched_linear(CheapProjection())
        layer.regional_adapters = {"one": self.adapter("lora")}
        shapes = []
        original = F.linear
        def counted(x, *args, **kwargs):
            shapes.append(tuple(x.shape))
            return original(x, *args, **kwargs)
        with patch.object(regional.F, "linear", side_effect=counted):
            layer(torch.randn(1, 12, 8))
        self.assertEqual(shapes, [(5, 8), (5, 3)])

    def test_dense_support_keeps_the_original_dense_path(self):
        for value in (1., .3):
            regional.CTX.token_plans = {}
            regional.CTX.token_masks = {"one": torch.full((1, 12, 1), value)}
            base = CheapProjection()
            layer = regional._patched_linear(base)
            layer.regional_adapters = {"one": self.adapter("lora")}
            x = torch.randn(2, 12, 8)
            output = layer(x)
            torch.testing.assert_close(output, self.reference(x, layer.regional_adapters, base))
            self.assertIsNone(regional._token_plan("one", regional.CTX.token_masks["one"], x, output).indices)

    def test_empty_unconditional_mask_skips_all_adapter_projections(self):
        regional.CTX.token_masks = {"one": torch.zeros(1, 12, 1)}
        layer = regional._patched_linear(CheapProjection())
        layer.regional_adapters = {"one": self.adapter("lora")}
        with patch.object(regional.F, "linear", side_effect=AssertionError("empty mask calculated")):
            self.assertFalse(layer(torch.randn(1, 12, 8)).any())
        self.assertEqual(layer.__dict__.get("_k2b_cast"), {})

    def test_strided_inputs_and_outputs_work_without_mutating_aliases(self):
        regional.CTX.token_masks = self.masks()
        base = AliasProjection()
        layer = regional._patched_linear(base)
        layer.regional_adapters = {"one": self.adapter("lora", out=8)}
        x = torch.randn(1, 8, 12).transpose(1, 2)
        untouched = x.clone()
        expected = self.reference(x, layer.regional_adapters, base)
        torch.testing.assert_close(layer(x), expected)
        self.assertTrue(torch.equal(x, untouched))

    def test_gradients_match_dense_reference(self):
        regional.CTX.token_masks = self.masks()
        base = CheapProjection()
        layer = regional._patched_linear(base)
        layer.regional_adapters = {uid: self.adapter("lora") for uid in ("one", "two")}
        x = torch.randn(1, 12, 8, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        layer(x).square().sum().backward()
        self.reference(reference_x, layer.regional_adapters, base).square().sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)

    def test_dense_and_sparse_adapters_can_mix_in_either_order_with_negative_strength(self):
        for order in (("one", "two"), ("two", "one")):
            with self.subTest(order=order):
                regional.CTX.token_plans = {}
                regional.CTX.token_masks = self.masks()
                regional.CTX.token_masks["one"] = torch.ones(1, 12, 1)
                base = CheapProjection()
                layer = regional._patched_linear(base)
                adapters = {"one": self.adapter("lora"), "two": self.adapter("lokr")}
                kind, a, b, scale = adapters["two"]
                adapters["two"] = (kind, a, b, -scale)
                layer.regional_adapters = {uid: adapters[uid] for uid in order}
                x = torch.randn(1, 12, 8, requires_grad=True)
                reference_x = x.detach().clone().requires_grad_()
                actual = layer(x)
                expected = self.reference(reference_x, layer.regional_adapters, base)
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, reference_x.grad)

    def test_unknown_layout_and_batch_mismatch_do_not_apply_adapter(self):
        regional.CTX.token_masks = self.masks(mixed=True)
        base = CheapProjection()
        layer = regional._patched_linear(base)
        layer.regional_adapters = {"one": self.adapter("lora")}
        for shape in ((1, 12, 8), (2, 11, 8)):
            self.assertTrue(torch.equal(layer(torch.randn(shape)), torch.zeros(*shape[:-1], 7)))


if __name__ == "__main__":
    unittest.main()
