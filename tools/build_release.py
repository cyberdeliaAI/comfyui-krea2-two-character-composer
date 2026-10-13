"""Build install and source ZIPs without caches or generated archives."""

from pathlib import Path
import re
import zipfile

root = Path(__file__).resolve().parents[1]
version = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.MULTILINE).group(1)
destination = root / "dist" / f"comfyui-krea2-two-character-composer-{version}.zip"
destination.parent.mkdir(exist_ok=True)
files = [root / name for name in ("__init__.py", "composer.py", "prompting.py", "negpip_bridge.py",
                                  "pyproject.toml", "LICENSE", "README.md", "CHANGELOG.md")]
for directory in ("web", "engine", "example_workflows", "assets"):
    files.extend(path for path in (root / directory).rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts)
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, Path("comfyui-krea2-two-character-composer") / path.relative_to(root))
print(destination)
source_destination = destination.with_name(destination.stem + "-source.zip")
source_files = files + [root / ".gitignore"]
for directory in ("tests", "tools", ".github"):
    source_files.extend(path for path in (root / directory).rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
with zipfile.ZipFile(source_destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(source_files):
        archive.write(path, Path("comfyui-krea2-two-character-composer") / path.relative_to(root))
print(source_destination)
