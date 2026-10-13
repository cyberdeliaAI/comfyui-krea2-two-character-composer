import json
from pathlib import Path
import unittest


class WorkflowTests(unittest.TestCase):
    def test_every_link_matches_both_endpoint_slots_and_types(self):
        path = Path(__file__).resolve().parents[1] / "example_workflows" / "two_characters.json"
        workflow = json.loads(path.read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(len(nodes), len(workflow["nodes"]))
        self.assertEqual(workflow["last_link_id"], len(workflow["links"]))
        for id, src, src_slot, dst, dst_slot, kind in workflow["links"]:
            output = nodes[src]["outputs"][src_slot]
            input = nodes[dst]["inputs"][dst_slot]
            self.assertIn(id, output["links"])
            self.assertEqual(input["link"], id)
            self.assertEqual(output["type"], input["type"])
            self.assertEqual(input["type"], kind)
        composer = next(n for n in nodes.values() if n["type"] == "Krea2TwoCharacterComposer")
        self.assertEqual(len(composer["widgets_values"]), 17)
        self.assertEqual(composer["widgets_values"][10:12], ["Left / Right", 60])
        self.assertEqual(composer["widgets_values"][4], "None")
        self.assertEqual(composer["widgets_values"][8], "None")
        self.assertEqual(composer["widgets_values"][12], "Balanced")
        self.assertEqual(composer["widgets_values"][13:], ["5, 10, 42, 85; 53, 10, 42, 85", 0, 1.0, "Auto"])
        self.assertEqual(composer["properties"]["krea2_composer_widget_layout"], 2)
        names = [o["name"] for o in composer["outputs"]]
        self.assertEqual(names, ["model", "positive", "latent", "region_masks", "negative", "prompts"])
        sampler = next(n for n in nodes.values() if n["type"] == "KSampler")
        negative_link = sampler["inputs"][2]["link"]
        self.assertIn(negative_link, composer["outputs"][4]["links"])
        clip_loader = next(n for n in nodes.values() if n["type"] == "CLIPLoader")
        self.assertEqual(clip_loader["widgets_values"][1], "krea2")
        self.assertEqual({node["type"] for node in nodes.values()}, {
            "UNETLoader", "CLIPLoader", "VAELoader", "EmptySD3LatentImage",
            "Krea2TwoCharacterComposer", "KSampler", "VAEDecode", "SaveImage", "Note",
        })


if __name__ == "__main__":
    unittest.main()
