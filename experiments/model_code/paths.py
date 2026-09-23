"""Portable defaults for source snapshots that consume historical artifacts.

These settings relocate defaults only. Paths inside sealed protocol manifests
remain part of their provenance and are not rewritten silently.
"""
from pathlib import Path
import os

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SOURCE_ROOT.parent


def _path(name, default):
    # Relative settings follow the checkout, even when invoked from another cwd.
    path = Path(os.path.expandvars(os.environ.get(name, str(default)))).expanduser()
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()


RESEARCH_ROOT = _path("BNG_RESEARCH_ROOT", PACKAGE_ROOT / "inputs" / "research")
OUTPUT_ROOT = _path(
    "BNG_OUTPUT_ROOT", RESEARCH_ROOT / "Residual_gain_regulization" / "outputs"
)
DATA_ROOT = _path("BNG_DATA_ROOT", PACKAGE_ROOT / "inputs" / "datasets")
MODEL_ROOT = _path("BNG_MODEL_ROOT", PACKAGE_ROOT / "models")
LLAMA_MODEL = _path("BNG_LLAMA_MODEL", MODEL_ROOT / "Meta-Llama-3-8B-Instruct")
CACHE_ROOT = _path("BNG_CACHE_ROOT", PACKAGE_ROOT / "build" / "cache")
SCRATCH_ROOT = _path("BNG_SCRATCH_ROOT", PACKAGE_ROOT / "build" / "scratch")
BLIP2_ASSET_ROOT = _path("BNG_BLIP2_ASSET_ROOT", PACKAGE_ROOT / "inputs" / "blip2")
