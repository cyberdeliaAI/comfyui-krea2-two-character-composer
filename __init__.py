from .composer import Krea2TwoCharacterComposer

NODE_CLASS_MAPPINGS = {"Krea2TwoCharacterComposer": Krea2TwoCharacterComposer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2TwoCharacterComposer": "Krea2 Two Character Composer"
}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
