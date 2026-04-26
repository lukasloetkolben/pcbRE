#!/usr/bin/env python3
"""Convenience launcher: `python run.py` from the repo root."""

import os
import sys

# Allow running this script directly without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcbre.app import main  # noqa: E402

if __name__ == "__main__":
    main()
