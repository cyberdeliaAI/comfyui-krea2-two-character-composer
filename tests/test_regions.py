"""0.4 layouts, masks, common prompts and legacy inline LoRA regressions."""

import importlib
import sys
import unittest
from unittest.mock import patch

import torch

from tests import helpers

helpers.load_package()
composer = importlib.import_module(f"{helpers.PACKAGE}.composer")
prompting = importlib.import_module(f"{helpers.PACKAGE}.prompting")
regional = importlib.import_module(f"{helpers.PACKAGE}.engine.regional")


class RegionTests(unittest.TestCase):
    def test_free_boxes_leave_background_and_snap_edges(self):
        masks, _ = composer.region_masks("Free boxes", 50, "0, 0, 40, 75; 60, 25, 40, 75",
                                          (None, None), 128, 96, 16)
        tokens = torch.stack([regional._latent_mask(m, 6, 8, "cpu") for m in masks])
        self.assertTrue((tokens > 0.5).any(1).all())
        self.assertTrue((~(tokens > 0.5).any(0)).any())
        self.assertTrue(((tokens == 0) | (tokens == 1)).all())

    def test_overlap_belongs_to_first_character(self):
        masks, _ = composer.region_masks("Free boxes", 50, "0, 0, 70, 100; 50, 0, 50, 100",
                                          (None, None), 128, 96, 16)
        original = torch.stack([regional._latent_mask(m, 6, 8, "cpu") for m in masks])
        exclusive = regional._exclusive_soft(original)
        overlap = (original == 1).all(0)
        self.assertTrue(overlap.any())
        self.assertTrue((exclusive[0, overlap] == 1).all())
        self.assertFalse(exclusive[1, overlap].any())

    def test_completely_hidden_second_character_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no area of its own"):
            composer.region_masks("Free boxes", 50, "0, 0, 100, 100; 10, 10, 30, 30",
                                  (None, None), 128, 96, 16)

    def test_mask_input_replaces_only_its_character(self):
        replacement = torch.zeros(96, 128); replacement[:32, :32] = 1
        masks, report = composer.region_masks("Left / Right", 50, prompting.DEFAULT_BOXES,
                                              (replacement, None), 128, 96, 16)
        torch.testing.assert_close(masks[0], replacement)
        self.assertEqual(masks[1].sum().item(), 96 * 64)
        self.assertIn("Character 1 from its mask input", report)

    def test_mask_batch_and_supported_singleton_shapes_use_first(self):
        first = torch.zeros(32, 32); first[:, :16] = 1
        batch = torch.stack([first, 1 - first])
        for shaped in (first, batch, batch[:, None], batch[..., None]):
            with self.subTest(shape=tuple(shaped.shape)):
                torch.testing.assert_close(composer.input_mask(shaped, 32, 32), first)

    def test_too_thin_faint_and_empty_masks_fail_on_token_grid(self):
        first = torch.zeros(96, 128); first[:, :32] = 1
        thin = torch.zeros(96, 128); thin[:, 100:101] = 1
        for bad in (thin, torch.full_like(first, 0.4), torch.zeros_like(first)):
            with self.subTest(maximum=bad.max().item()), self.assertRaisesRegex(ValueError, "empty or too thin"):
                composer.check_token_coverage([first, bad], 128, 96, 16)

    def test_mask_centroids_determine_actual_order(self):
        first = torch.zeros(96, 128); first[:, 96:] = 1
        second = torch.zeros(96, 128); second[:, :32] = 1
        sides = prompting.relative_sides(composer.centroid(first), composer.centroid(second))
        self.assertEqual(sides, ("On the right side of the image", "On the left side of the image"))

    def test_invalid_box_coordinates_are_rejected(self):
        for text in ("", "0,0,30,30", "0,0,0,30; 50,0,40,30", "90,0,20,30; 0,0,40,30"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prompting.parse_boxes(text)


class CommonPromptTests(unittest.TestCase):
    def test_definitions_work_across_fields_and_before_definition(self):
        texts, unknown = prompting.expand_common(["{outfit}", "{outfit: navy uniform}", "a knight, {OUTFIT}"])
        self.assertEqual(texts, ["navy uniform", "navy uniform", "a knight, navy uniform"])
        self.assertEqual(unknown, [])

    def test_unknown_names_choices_and_numeric_braces_remain_text(self):
        texts, unknown = prompting.expand_common(["{outfit: navy uniform}", "{missing}, {a|b}, {word:1.2}"])
        self.assertEqual(texts[1], "{missing}, {a|b}, {word:1.2}")
        self.assertEqual(unknown, ["{missing}"])
        self.assertEqual(prompting.expand_common(["literal {missing}, {a|b}"])[0], ["literal {missing}, {a|b}"])

    def test_conflicting_definitions_raise(self):
        with self.assertRaisesRegex(ValueError, "defined twice"):
            prompting.expand_common(["{outfit: red}", "{outfit: blue}"])


class InlineLoraTests(unittest.TestCase):
    def test_tags_are_loaded_and_removed_from_encoder_text(self):
        composer._LORA_CACHE.clear()
        sys.modules["comfy.utils"].loaded.clear()
        clip = helpers.FakeClip()
        result = composer.Krea2TwoCharacterComposer().compose(
            helpers.FakePatcher(), clip, helpers.latent(), "Left / Right", 60, "garden", "",
            "", "<lora:characters/a.safetensors:0.75>, a knight", "None", 1.0,
            "", "a wizard", "None", 1.0, "Balanced")
        self.assertNotIn("<lora:", clip.encoded[1])
        self.assertEqual(sys.modules["comfy.utils"].loaded, ["/loras/characters/a.safetensors"])
        self.assertIn("characters/a.safetensors @ 0.75", result[-1])

    def test_zero_negative_weight_does_not_enable_negpip(self):
        self.assertFalse(prompting.has_negative_weights("(beard:-0.0)"))
        clip = helpers.FakeClip()
        with patch.object(composer.negpip_bridge, "require", side_effect=AssertionError("must not load NegPiP")):
            composer.Krea2TwoCharacterComposer().compose(
                helpers.FakePatcher(), clip, helpers.latent(), "Left / Right", 60, "garden", "",
                "", "a knight, (beard:-0.0)", "None", 1.0,
                "", "a wizard", "None", 1.0, "Balanced")

    def test_ambiguous_tag_names_are_errors(self):
        with patch.object(sys.modules["folder_paths"], "get_filename_list", return_value=["left/a.safetensors", "right/a.safetensors"]):
            with self.assertRaisesRegex(ValueError, "matches several"):
                composer.resolve_lora_tag("a")

    def test_menu_takes_precedence_over_duplicate_tag(self):
        rows = composer.character_loras(("characters/a.safetensors", 0.75), [("a", 1.25)])
        self.assertEqual(rows, [("characters/a.safetensors", 0.75)])
