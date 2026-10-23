"""Benchmark implementations.

Registration is centralized in ``_registry`` (one row per benchmark).
Importing this package fires those ``register(...)`` calls.
"""

from . import _registry  # noqa: F401
