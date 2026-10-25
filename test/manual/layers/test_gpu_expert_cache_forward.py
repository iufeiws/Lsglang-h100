"""GPU expert cache forward on H100: CUTLASS slot hits plus UVA misses = dense reference.

Needs CUDA (SM90) and FlashInfer:
  python -m pytest -q test/manual/layers/test_gpu_expert_cache_forward.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="FlashInfer SM90 W4A16 MoE needs an H100",
)

_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _dequant(packed, scales):
    lo = _E2M1[(packed & 0xF).long()]
    hi = _E2M1[(packed >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(-2)
    return w * torch.pow(2.0, scales.float() - 127).repeat_interleave(32, dim=-1)


def _reference(x, ids, weights, w13, s13, w2, s2, limit):
    """Checkpoint layout: w13 gate rows first; DSv4 clamped SwiGLU."""
    rows, k = ids.shape
    I = w13.shape[1] // 2
    out = torch.zeros(rows, x.shape[1])
    for r in range(rows):
        for j in range(k):
            e = int(ids[r, j])
            if e < 0:
                continue
            h = _dequant(w13[e], s13[e]) @ x[r].float()
            gate, up = h[:I].clamp(max=limit), h[I:].clamp(-limit, limit)
            act = (up * gate * torch.sigmoid(gate)).bfloat16().float()
            out[r] += weights[r, j] * (_dequant(w2[e], s2[e]) @ act)
    return out


E, H, I, K, ROWS, S, P, LIMIT = 12, 512, 256, 4, 9, 6, 2, 10.0


# Registered host memory is never unregistered: a cache built on addresses a freed
# cache had registered fails with CUDA error 712, so every cache here stays alive.
_alive = []


@pytest.fixture(scope="module", params=["bf16", "fp8"])
def setup(request):
    """One cache per precision, kept for the whole process."""
    from sglang.srt.layers.moe.gpu_expert_cache import ForwardBuffers, LayerCache, MoeParams

    torch.manual_seed(3)
    experts = (
        torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8),
        torch.randint(119, 123, (E, 2 * I, H // 32), dtype=torch.uint8),
        torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8),
        torch.randint(119, 123, (E, H, I // 32), dtype=torch.uint8),
    )
    fp8 = request.param == "fp8"
    cache = LayerCache(layer_id=0, host_tensors=experts, num_slots=S, num_staging=P, fp8=fp8,
                       device="cuda")
    buffers = ForwardBuffers(num_experts=E, max_rows=ROWS, top_k=K, hidden=H,
                             intermediate=I, num_staging=P, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((S + P,), LIMIT, device="cuda"), limit=LIMIT,
                       tp_size=1, tp_rank=0,
                       fc2_act_scale=torch.ones((), device="cuda") if fp8 else None)
    _alive.append((experts, cache, buffers, params))
    return experts, cache, buffers, params


def _close(actual, expected, fp8):
    if fp8:
        # W4A8 quantizes activations to FP8 and clamps scale ranges: compare in aggregate.
        cos = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert cos > 0.99, float(cos)
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-2,
                                   atol=3e-2 * expected.abs().max().item())


def _routes():
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(ROWS)]).int()
    ids[::2, -1] = -1  # EP-remote routes
    return ids, torch.randn(ROWS, H).bfloat16(), torch.rand(ROWS, K)


def _run(cache, buffers, params, ids, x, weights, graph):
    from sglang.srt.layers.moe.gpu_expert_cache import cache_forward

    out = cache_forward(cache, hidden_states=x.cuda(), topk_ids=ids.cuda(),
                        topk_weights=weights.cuda(), buffers=buffers, params=params,
                        graph=graph)
    torch.cuda.synchronize()
    return out.cpu()


@pytest.mark.parametrize("graph", [False, True])
def test_hits_staged_and_overflow_misses_match_the_reference(setup, graph):
    experts, cache, buffers, params = setup
    ids, x, weights = _routes()
    expected = _reference(x, ids, weights, *experts, LIMIT)
    hist0 = cache.hist.clone()
    slot_of = cache.slot_of.clone()
    # Cold start holds experts 0..S-1; the other E-S > P experts are staged or overflow.
    assert len(set(ids[(ids >= S)].tolist())) > P
    _close(_run(cache, buffers, params, ids, x, weights, graph), expected, cache.fp8)
    local = ids.flatten()[ids.flatten() >= 0].long()
    assert torch.equal((cache.hist - hist0).cpu(), torch.bincount(local, minlength=E).int())
    # With every slot unmapped all routes are misses: the paths must agree.
    cache.slot_of.fill_(-1)
    try:
        _close(_run(cache, buffers, params, ids, x, weights, graph), expected, cache.fp8)
    finally:
        cache.slot_of.copy_(slot_of)


def test_prefetched_experts_are_served_without_a_copy(setup):
    """A prefetch with layer L+1's router stages its experts; the forward then copies
    nothing for them and still matches the reference."""
    from sglang.kernels.ops.moe.lk_verify_uva import (
        STAT_PREFETCH_USED,
        STAT_PREFETCHED,
        STAT_STAGE_COPIED,
        STAT_STAGE_KEPT,
    )
    from sglang.srt.layers.moe.gpu_expert_cache import (
        PrefetchTarget,
        _prefetch_targets,
        prefetch_next_layer,
        register_prefetch_target,
    )

    experts, cache, buffers, params = setup
    cache.stage_expert.fill_(-1)  # earlier tests may have left these experts staged
    cache.row_pending.zero_()
    # Two non-slotted experts that a router rigged to pick them predicts for every row.
    wanted = [S, S + 1]
    gate = torch.zeros(E, H, device="cuda")
    bias = torch.zeros(E, device="cuda")
    bias[wanted] = 10.0
    # Target key = the cache's layer id (0), so its forward is also the last prefetched one.
    register_prefetch_target(cache.layer_id, PrefetchTarget(
        cache=cache, gate_weight=gate, correction_bias=bias, scoring_func="sqrtsoftplus",
        top_k=2, ep_rank=0, programs=8, max_experts=P))
    try:
        ids, x, weights = _routes()
        ids[:, :2] = torch.tensor(wanted, dtype=torch.int32)
        ids[:, 2:] = torch.randint(0, S, (ROWS, K - 2), dtype=torch.int32)
        ids[:, 3] = torch.where(ids[:, 3] == ids[:, 2], (ids[:, 2] + 1) % S, ids[:, 3])
        expected = _reference(x, ids, weights, *experts, LIMIT)
        stats0 = cache.stats.clone()
        prefetch_next_layer(cache.layer_id - 1, x.cuda(), buffers)
        assert cache.prefetch_issued and buffers.prefetch_outstanding
        out = _run(cache, buffers, params, ids, x, weights, graph=True)
        assert not cache.prefetch_issued and not buffers.prefetch_outstanding
        delta = (cache.stats - stats0).cpu()
        assert int(delta[STAT_PREFETCHED]) == 2 and int(delta[STAT_PREFETCH_USED]) == 2
        assert int(delta[STAT_STAGE_KEPT]) == 2 and int(delta[STAT_STAGE_COPIED]) == 0
        _close(out, expected, cache.fp8)
    finally:
        _prefetch_targets.pop(cache.layer_id)
