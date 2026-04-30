"""Pytest configuration for the MLX rewrite track.

Keeping configuration minimal here: we rely on pyproject.toml's
``[tool.pytest.ini_options]`` for discovery, and on ``_helpers`` for
reusable helpers. This file only adjusts sys.path so ``_helpers`` can be
imported by sibling test modules without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
