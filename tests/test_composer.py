"""The composer node end to end: real engine, fake CLIP/model/NegPiP pack."""

import importlib
import math
import sys
import types
import unittest

import torch

from tests import helpers

package = helpers.load_package()
composer = importlib.import_module(f"{helpers.PACKAGE}.composer")
regional = importlib.import_module(f"{helpers.PACKAGE}.engine.regional")
bridge = importlib.import_module(f"{helpers.PACKAGE}.negpip_bridge")

WIDGETS = [
    "base_prompt", "negative_prompt",
    "character_1_trigger", "character_1_prompt", "character_1_lora", "character_1_strength",
    "character_2_trigger", "character_2_prompt", "character_2_lora", "character_2_strength",
    "direction", "character_1_percent", "separation",
]


def hide_installed_negpip(test):
    """Keep a real NegPiP pack loaded by other tests out of these tests."""
    found = {k: v for k, v in sys.modules.items()
             if k.endswith(".src.nodes") and hasattr(v, "NegPipPrompt")}
    for name in found:
        del sys.modules[name]
    test.addCleanup(sys.modules.update, found)


class FakeNegPipPack:
    """Registers modules shaped like comfyui-negpip-zimage in sys.modules."""

    PREFIX = "C:/ComfyUI/custom_nodes/comfyui-negpip-zimage-main"
    POS, SRC, SIDE = ("krea2_negpip_negative_positions", "krea2_negpip_source_length",
                      "krea2_negpip_sidecar_tokens")

    def __init__(self):
        self.merged, self.patched, self.clips = [], [], []
        pack = self

        class NegClip(helpers.FakeClip):
            def encode_from_tokens_scheduled(self, tokens):
                [[cond, extra]] = super().encode_from_tokens_scheduled(tokens)
                if composer.has_negative_weights(tokens):
                    n = cond.shape[1]
                    cond = torch.cat([cond, torch.zeros(1, 1, cond.shape[2])], dim=1)
                    extra.update({pack.POS: [0], pack.SRC: n, pack.SIDE: 1,
                                  "attention_mask": torch.ones(1, n + 1)})
                return [[cond, extra]]

        def patch(model, clip):
            pack.patched.append(model)
            m = model.clone()
            m.model_options["transformer_options"]["krea2_negpip"] = {"enabled": True}
            pack.clips.append(NegClip())
            return m, pack.clips[-1]

        def merge(positive, negative):
            pack.merged.append((positive, negative))
            return f"{positive}, ({negative}:-1)"

        nodes = types.ModuleType(self.PREFIX + ".src.nodes")
        nodes.NegPipPrompt, nodes.merge_prompts = object, merge
        krea2 = types.ModuleType(self.PREFIX + ".src.negpip_krea2")
        krea2.patch_krea2_negpip = patch
        krea2.NEGATIVE_POSITIONS_EXTRA_KEY = self.POS
        self.modules = {nodes.__name__: nodes, krea2.__name__: krea2}

    def __enter__(self):
        sys.modules.update(self.modules)
        return self

    def __exit__(self, *exc):
        for name in self.modules:
            sys.modules.pop(name, None)


class ComposerTests(unittest.TestCase):
    def setUp(self):
        hide_installed_negpip(self)
        composer._LORA_CACHE.clear()
        sys.modules["comfy.utils"].loaded.clear()
        self.clip = helpers.FakeClip()
        self.args = dict(
            model=helpers.FakePatcher(), clip=self.clip, latent=helpers.latent(),
            direction="Left / Right", character_1_percent=60,
            base_prompt="Two people in a garden, daylight.", negative_prompt="",
            character_1_trigger="xk3char", character_1_prompt="a red knight, full armor",
            character_1_lora="characters/a.safetensors", character_1_strength=0.75,
            character_2_trigger="m_wiz", character_2_prompt="a blue wizard",
            character_2_lora="b.safetensors", character_2_strength=1.25,
            separation="Balanced",
        )

    def compose(self, **overrides):
        return composer.Krea2TwoCharacterComposer().compose(**(self.args | overrides))

    def bundle(self, model):
        return model.model_options["transformer_options"][regional.BUNDLE_KEY]

    def test_schema_order_matches_the_frontend(self):
        required = list(composer.Krea2TwoCharacterComposer.INPUT_TYPES()["required"])
        self.assertEqual(required, ["model", "clip", "latent"] + WIDGETS)
        js = (helpers.ROOT / "web" / "composer.js").read_text()
        self.assertIn("const ADDED =", js)
        optional = composer.Krea2TwoCharacterComposer.INPUT_TYPES()["optional"]
        self.assertEqual(set(optional), {"boxes", "lora_edge_softness", "base_strength",
                                         "character_1_mask", "character_2_mask", "preview_ratio", "width", "height"})
        self.assertEqual(composer.Krea2TwoCharacterComposer.RETURN_NAMES[:4],
                         ("model", "positive", "latent", "region_masks"))

    def test_plain_run_builds_prompts_masks_and_routes_loras(self):
        model, positive, latent, masks, negative, report = self.compose()
        scene, first, second = self.clip.encoded
        self.assertNotIn("xk3char", scene)
        self.assertIn("On the left side of the image: a red knight.", scene)
        self.assertIn("On the right side of the image: a blue wizard.", scene)
        self.assertEqual(first, "xk3char, a red knight, full armor")
        self.assertEqual(second, "m_wiz, a blue wizard")
        self.assertIs(latent, self.args["latent"])
        self.assertEqual(masks.shape, (2, 96, 128))
        self.assertTrue(torch.equal(masks.sum(0), torch.ones(96, 128)))
        # 60% of 128px = 76.8px, snapped to the 16px token grid = 80px
        self.assertEqual(int(masks[0].sum()), 80 * 96)
        bundle = self.bundle(model)
        lengths = [len(t.split()) for t in self.clip.encoded]
        self.assertEqual(bundle["txt_total"], sum(lengths))
        self.assertEqual([s["seg"] for s in bundle["loras"]], [1, 2])
        self.assertEqual(sys.modules["comfy.utils"].loaded[-2:],
                         ["/loras/characters/a.safetensors", "/loras/b.safetensors"])
        self.assertEqual(negative[0][0].shape, positive[0][0].shape)
        self.assertFalse(negative[0][0].any())
        self.assertIn("NegPiP: off", report)
        self.assertFalse(bundle["restrict_img_attn"])

    def test_preview_controls_do_not_resize_generation_or_regions(self):
        _, _, latent, masks, _, _ = self.compose(preview_ratio="9:16", width=1920, height=1080)
        self.assertIs(latent, self.args["latent"])
        self.assertEqual(masks.shape, (2, 96, 128))
        self.assertEqual(int(masks[0].sum()), 80 * 96)

    def test_blank_free_boxes_use_default_regions_for_api_workflows(self):
        for boxes in ("", "   ", None):
            with self.subTest(boxes=boxes):
                _, _, _, masks, _, report = self.compose(direction="Free boxes", boxes=boxes)
                expected, _ = composer.region_masks("Free boxes", 60, composer.DEFAULT_BOXES,
                                                    (None, None), 128, 96, 16)
                torch.testing.assert_close(masks, torch.stack(expected))
                self.assertIn(composer.DEFAULT_BOXES, report)

    def test_passive_metadata_matches_encoded_regions_and_does_not_change_tensors(self):
        import json
        from unittest.mock import patch
        original = regional.apply_regional
        received = []
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            received.append(result)
            return result
        with patch.object(regional, "apply_regional", side_effect=capture):
            result = self.compose()
        positive, negative = result[1], result[4]
        metadata = positive[0][1][composer.METADATA_KEY]
        self.assertIs(positive[0][0], received[0][1][0][0])
        self.assertNotIn(composer.METADATA_KEY, negative[0][1])
        self.assertEqual(metadata["scene_prompt"], self.clip.encoded[0])
        self.assertEqual(metadata["character_prompts"], self.clip.encoded[1:])
        self.assertEqual(metadata["negative_prompt"], "")
        self.assertEqual(metadata["loras"][0], [{"name": "characters/a.safetensors", "strength": .75}])
        self.assertEqual((metadata["width"], metadata["height"]), (128, 96))
        json.dumps(metadata)  # payload has no tensors/model objects

    def test_metadata_keeps_expanded_common_prompts_and_separate_negpip_negative(self):
        with FakeNegPipPack():
            _, positive, _, _, _, _ = self.compose(
                base_prompt="a garden, {outfit: navy uniform}",
                character_1_prompt="a knight wearing {outfit}", negative_prompt="blurry")
        metadata = positive[0][1][composer.METADATA_KEY]
        self.assertIn("navy uniform", metadata["character_prompts"][0])
        self.assertNotIn("{outfit}", metadata["scene_prompt"])
        self.assertEqual(metadata["negative_prompt"], "blurry")
        self.assertTrue(metadata["negpip"])
        self.assertNotEqual(metadata["scene_prompt"], metadata["encoded_scene_prompt"])

    def test_strong_and_none_loras(self):
        model, *_ = self.compose(separation="Strong", character_1_lora="None", character_2_strength=0)
        bundle = self.bundle(model)
        self.assertTrue(bundle["restrict_img_attn"])
        self.assertEqual(bundle["restrict_end_sigma"], 0.5)
        self.assertEqual(bundle["loras"], [])

    def test_negative_prompt_uses_negpip_and_accounts_for_sidecars(self):
        with FakeNegPipPack() as pack:
            model, positive, _, _, negative, report = self.compose(negative_prompt="blurry, text")
        self.assertEqual(len(pack.merged), 1)
        self.assertIn("On the left side", pack.merged[0][0])
        # only the scene went through NegPiP's CLIP; the characters stay stock
        self.assertEqual(len(pack.clips[0].encoded), 1)
        self.assertEqual(self.clip.encoded, ["xk3char, a red knight, full armor", "m_wiz, a blue wizard"])
        self.assertEqual(pack.merged[0][1], "blurry, text")
        bundle = self.bundle(model)
        self.assertTrue(bundle["expects_negpip"])
        self.assertEqual(bundle["txt_total_with_sidecars"], bundle["txt_total"] + 1)
        self.assertEqual(positive[0][0].shape[1], bundle["txt_total"] + 1)
        self.assertEqual(model.model_options["transformer_options"]["krea2_negpip"], {"enabled": True})
        for key in (FakeNegPipPack.POS, FakeNegPipPack.SRC, FakeNegPipPack.SIDE, "attention_mask"):
            self.assertNotIn(key, positive[0][1])
            self.assertNotIn(key, negative[0][1])
        self.assertIn("NegPiP: on", report)

    def test_inline_negative_weight_in_one_character_enables_negpip(self):
        with FakeNegPipPack() as pack:
            model, *_ = self.compose(character_2_prompt="a blue wizard, (beard:-0.6)")
        self.assertEqual(pack.merged, [])           # no global negative
        self.assertEqual(len(pack.patched), 1)
        self.assertEqual(pack.clips[0].encoded, ["m_wiz, a blue wizard, (beard:-0.6)"])
        self.assertEqual(len(self.clip.encoded), 2)  # scene and character 1: stock CLIP
        segments = self.bundle(model)["segments"]
        total = self.bundle(model)["txt_total"]
        self.assertEqual(segments[2][1], total)     # character 2 ends the stripped sequence
        self.assertEqual(self.bundle(model)["txt_total_with_sidecars"], total + 1)

    def test_negatives_without_negpip_explain_how_to_install_it(self):
        with self.assertRaisesRegex(bridge.NegPipUnavailable, "comfyui-negpip-zimage"):
            self.compose(negative_prompt="blurry")

    def test_bad_inputs_fail_early(self):
        cases = [
            dict(character_1_trigger="", character_1_prompt="  "),
            dict(character_2_lora="missing.safetensors"),
            dict(character_1_strength=math.inf),
            dict(character_2_strength=math.nan),
            dict(separation="Unknown"),
            dict(direction="Diagonal"),
            dict(character_1_percent=0),
            dict(latent={"samples": torch.zeros(1, 4, 12, 16)}),
            dict(latent={"samples": torch.zeros(1, 16, 3, 12, 16)}),
        ]
        for overrides in cases:
            with self.subTest(overrides=list(overrides)), self.assertRaises(ValueError):
                self.compose(**overrides)

    def test_non_krea2_models_are_rejected(self):
        other = helpers.FakePatcher()
        other.model = types.SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "Connect a Krea 2 model"):
            self.compose(model=other)

    def test_split_lands_on_token_boundaries_for_every_percentage(self):
        left = torch.zeros(96, 128)
        for direction, size in (("Left / Right", 128), ("Top / Bottom", 96)):
            for percent in range(1, 100):
                *_, masks, _, _ = self.compose(direction=direction, character_1_percent=percent)
                axis = 1 if direction == "Left / Right" else 0
                first = int(masks[0].sum(dim=1 - axis).count_nonzero())
                self.assertEqual(first % 16, 0)
                self.assertTrue(16 <= first <= size - 16)
                self.assertTrue(torch.equal(masks.sum(0), torch.ones_like(left)))
                nearest = round(size * percent / 100 / 16) * 16
                self.assertEqual(first, min(max(nearest, 16), size - 16))

    def test_trigger_only_character_is_allowed(self):
        self.compose(character_2_prompt="")
        self.assertEqual(self.clip.encoded[2], "m_wiz")
        self.assertIn("On the right side of the image: a character.", self.clip.encoded[0])


class BridgeTests(unittest.TestCase):
    def setUp(self):
        hide_installed_negpip(self)

    def test_find_returns_none_without_the_pack_and_uses_pack_keys(self):
        self.assertIsNone(bridge.find())
        with FakeNegPipPack():
            np = bridge.find()
        self.assertEqual(np.metadata_keys, (FakeNegPipPack.POS, FakeNegPipPack.SRC, FakeNegPipPack.SIDE))

    def test_inconsistent_sidecar_metadata_is_an_error(self):
        with FakeNegPipPack():
            np = bridge.find()
        cond = torch.zeros(1, 6, 4)
        self.assertEqual(np.sidecar_tokens(cond, {}, expect_negative=False), 0)
        self.assertEqual(np.sidecar_tokens(cond, {np.positions_key: [1], np.source_length_key: 5,
                                                  np.sidecar_tokens_key: 1}, True), 1)
        with self.assertRaises(RuntimeError):
            np.sidecar_tokens(cond, {np.positions_key: [1], np.source_length_key: 4,
                                     np.sidecar_tokens_key: 1}, True)
        with self.assertRaisesRegex(RuntimeError, "position data was lost"):
            np.sidecar_tokens(cond, {}, expect_negative=True)


if __name__ == "__main__":
    unittest.main()
