"""Residency decisions of the GPU expert cache (pure CPU, no GPU needed).

  python -m pytest -q test/manual/layers/test_gpu_expert_cache_policy.py
"""

import torch

from sglang.srt.layers.moe.gpu_expert_cache import plan_swaps


def test_empty_slots_take_the_hottest_experts_first():
    scores = torch.tensor([0.0, 5.0, 1.0, 9.0, 3.0])
    swaps = plan_swaps(scores, [-1, 2, -1], budget=8, min_gain=1.5)
    assert swaps[:2] == [(0, -1, 3), (2, -1, 1)]
    # The cold resident (expert 2, score 1) then yields to expert 4 (3 > 1.5 * 1).
    assert swaps[2:] == [(1, 2, 4)]


def test_budget_and_gain_limit_the_swaps():
    scores = torch.tensor([10.0, 9.0, 8.0, 1.0, 1.0, 1.0])
    owner = [3, 4, 5]
    assert len(plan_swaps(scores, owner, budget=2, min_gain=1.5)) == 2
    # 10 > 1.5 * 6 replaces the coldest resident; 9 is not above 1.5 * 7, so it stops.
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 7.0, 6.0])
    swaps = plan_swaps(scores, owner, budget=8, min_gain=1.5)
    assert swaps == [(2, 5, 0)]


def test_never_swaps_in_a_resident_or_unrouted_expert():
    scores = torch.tensor([4.0, 0.0, 0.0, 2.0])
    swaps = plan_swaps(scores, [0, 3], budget=4, min_gain=1.0)
    assert swaps == []
    swaps = plan_swaps(torch.zeros(4), [-1, -1], budget=4, min_gain=1.0)
    assert swaps == []
