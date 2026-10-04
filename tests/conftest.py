import sys
from pathlib import Path

# The modules in src/ import each other by bare name (`import gltf`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
