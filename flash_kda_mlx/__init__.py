"""FlashKDA — MLX rewrite track (work in progress).

Public API mirrors ``flash_kda.fwd`` so existing tests and fixtures can be
reused with minimal translation. See plan.md §Phase 7 for the API
stabilization discussion.

.. warning::
    MLX v1 is in active development. See STATUS.md for current scope.
"""

from __future__ import annotations

from .ops import Backend, FwdResult, fwd


__all__ = ["fwd", "FwdResult", "Backend"]
