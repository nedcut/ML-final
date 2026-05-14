"""Run the CORAL training module from the repository root."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

runpy.run_module("src.train_coral", run_name="__main__")
