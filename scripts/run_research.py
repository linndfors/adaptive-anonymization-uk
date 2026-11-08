#!/usr/bin/env python3
"""Run the installed CLI directly from a source checkout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anonrisk_uk.cli import main

if __name__ == "__main__":
    main()
