"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
MOE V2 - python package entry: dynamic CPU-backend loading.
"""
from ._version import __version__, __version_tuple__

from ._dynamic_loader import get_moe_classes, _HELPERS

_classes = get_moe_classes()

MOEConfigV2 = _classes["MOEConfigV2"]

# BF16 activation
MOE_BF16  = _classes.get("MOE_BF16")
MOE_FP8   = _classes.get("MOE_FP8")
MOE_WNA16 = _classes.get("MOE_WNA16")
MOE_NVFP4 = _classes.get("MOE_NVFP4")
MOE_MXFP4 = _classes.get("MOE_MXFP4")

# FP16 activation
MOE_FP16       = _classes.get("MOE_FP16")
MOE_FP8_FP16   = _classes.get("MOE_FP8_FP16")
MOE_WNA16_FP16 = _classes.get("MOE_WNA16_FP16")
MOE_NVFP4_FP16 = _classes.get("MOE_NVFP4_FP16")
MOE_MXFP4_FP16 = _classes.get("MOE_MXFP4_FP16")

_helpers = {name: _classes.get(name) for name in _HELPERS}
globals().update(_helpers)

__all__ = [
    "__version__",
    "__version_tuple__",
    "MOEConfigV2",
    "MOE_BF16",
    "MOE_FP8",
    "MOE_WNA16",
    "MOE_NVFP4",
    "MOE_MXFP4",
    "MOE_FP16",
    "MOE_FP8_FP16",
    "MOE_WNA16_FP16",
    "MOE_NVFP4_FP16",
    "MOE_MXFP4_FP16",
    *_HELPERS,
]