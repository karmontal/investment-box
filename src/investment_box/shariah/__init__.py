"""Shariah compliance layer.

``constraints`` is available from Phase 1 because the hard constraints must be
enforceable before any order path exists. Screening providers, purification and
zakat arrive in Phase 3.
"""

from investment_box.shariah.constraints import (
    HardConstraints,
    assert_order_permissible,
    is_forbidden_instrument,
)

__all__ = ["HardConstraints", "assert_order_permissible", "is_forbidden_instrument"]
