import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "prompting_under_test", Path(__file__).resolve().parents[1] / "prompting.py")
prompting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompting)


class SplitTests(unittest.TestCase):
    def test_all_percentages_cover_exactly_two_complementary_regions(self):
        for direction in prompting.DIRECTIONS[:2]:
            for percent in range(1, 100):
                first, second = prompting.split_regions(direction, percent)
                self.assertAlmostEqual(first["w"] * first["h"], percent / 100)
                self.assertAlmostEqual(second["w"] * second["h"], 1 - percent / 100)
                axis, extent = ("x", "w") if direction == "Left / Right" else ("y", "h")
                self.assertEqual(first[axis] + first[extent], second[axis])
                self.assertAlmostEqual(second[axis] + second[extent], 1)

    def test_invalid_values_fail(self):
        for value in (0, 100, -1, math.nan, math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prompting.split_regions("Left / Right", value)
        with self.assertRaises(ValueError):
            prompting.split_regions("Diagonal", 50)


class PromptTests(unittest.TestCase):
    def test_extreme_split_still_names_the_chosen_sides_without_size_words(self):
        scene = prompting.scene_prompt("Two people in a garden, daylight.",
                                       prompting.relative_sides((0.25, 0.5), (0.75, 0.5)),
                                       ["a man with spiky black hair", "a woman in a red coat"])
        self.assertEqual(
            scene,
            "Two people in a garden, daylight. Two distinct characters in one coherent scene. "
            "On the left side of the image: a man with spiky black hair. "
            "On the right side of the image: a woman in a red coat.")
        for word in ("center", "dominating", "prominent", "Character 1"):
            self.assertNotIn(word, scene)
        vertical = prompting.scene_prompt("", prompting.relative_sides((0.5, 0.25), (0.5, 0.75)), ["a", "b"])
        self.assertTrue(vertical.startswith("Two distinct characters"))
        self.assertIn("At the top of the image: a.", vertical)
        self.assertIn("At the bottom of the image: b.", vertical)

    def test_trigger_goes_first_in_the_region_and_never_into_the_hint(self):
        self.assertEqual(prompting.region_prompt("ohwx", "woman with red hair, smiling"),
                         "ohwx, woman with red hair, smiling")
        # already present: not repeated
        self.assertEqual(prompting.region_prompt("ohwx", "ohwx woman with red hair, smiling"),
                         "ohwx woman with red hair, smiling")
        self.assertEqual(prompting.hint_subject("ohwx woman with red hair, smiling", "ohwx"),
                         "woman with red hair")
        self.assertEqual(prompting.region_prompt("a1, b_2\n", ""), "a1, b_2")
        self.assertEqual(prompting.region_prompt("", "  just text "), "just text")

    def test_legacy_inline_trigger_clause_is_skipped_in_the_hint(self):
        self.assertEqual(prompting.hint_subject("trigger_character_1, a man with a yellow cap"),
                         "a man with a yellow cap")
        self.assertEqual(prompting.hint_subject("cybexpr1"), "a character")
        self.assertEqual(prompting.hint_subject(""), "a character")

    def test_hints_drop_weights_negatives_and_lora_tags(self):
        text = "(a tall knight:1.2), (beard:-0.6) <lora:knight:0.8>, silver armor"
        self.assertEqual(prompting.plain_text(text), "a tall knight, silver armor")
        self.assertEqual(prompting.hint_subject(text), "a tall knight")
        self.assertEqual(prompting.hint_subject("(beard:-1), a monk"), "a monk")

    def test_negative_weight_detection(self):
        self.assertTrue(prompting.has_negative_weights("a man, (beard:-0.6)"))
        self.assertTrue(prompting.has_negative_weights("(beard: -1)"))
        self.assertTrue(prompting.has_negative_weights("(beard:-.5)"))
        self.assertFalse(prompting.has_negative_weights("(beard:- 1)"))  # literal for ComfyUI
        self.assertFalse(prompting.has_negative_weights("(beard:0.6), 16:9, 12:30"))
        self.assertFalse(prompting.has_negative_weights(""))

    def test_long_subjects_are_cut_on_a_word_boundary(self):
        subject = prompting.hint_subject("a " + "very " * 40 + "long description")
        self.assertLessEqual(len(subject), 90)
        self.assertFalse(subject.endswith(" "))
        self.assertTrue(subject.endswith("very"))


if __name__ == "__main__":
    unittest.main()
