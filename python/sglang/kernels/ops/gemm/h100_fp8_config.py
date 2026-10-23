"""Bounded H100 block32 launch profiles, not a claim of measured autotuning.

Exact per-device/per-shape tuned files take precedence in fp8_kernel.py.
These seed profiles cover the V4.1 projection shapes and small token counts;
large prefill, other block sizes, and other devices retain their old choices.
"""

import functools
import json
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _profiles():
    path = Path(__file__).with_name("configs") / "h100_block_fp8_small_m.json"
    return json.loads(path.read_text())


def get_h100_block_fp8_config(M, N, K, block_n, block_k, device_name):
    if not device_name.replace(" ", "_").startswith("NVIDIA_H100"):
        return None
    if (block_n, block_k) != (32, 32) or not 1 <= M <= 96:
        return None
    profiles = _profiles()
    max_splits = profiles["shapes"].get(f"{N},{K}")
    if max_splits is None:
        return None
    bucket = next(b for b in profiles["buckets"] if M <= b["max_m"])
    config = dict(profiles["tile"])
    config["BLOCK_SIZE_M"] = bucket["block_m"]
    config["SPLIT_K"] = max(1, max_splits // bucket["split_divisor"])
    return config
