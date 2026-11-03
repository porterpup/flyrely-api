"""Feature code shared by training (vectorised backtests) and serving (single row).

Every group exposes a vectorised function and a ``serve_row`` that runs the SAME
function on a one-row input, so the two cannot drift apart.
"""

from . import booking, network, recent, rotation, weather  # noqa: F401
