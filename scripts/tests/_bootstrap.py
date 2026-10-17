"""Shared repository paths for unittest discovery from scripts/tests."""

from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
