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


E, H, I, K, ROWS, R, C, LIMIT = 12, 512, 256, 4, 9, 6, 2, 10.0

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
    cache = LayerCache(layer_id=0, host_tensors=experts, num_rows=R, max_copy=C,
                       score_decay=0.99, fp8=fp8, device="cuda")
    buffers = ForwardBuffers(num_experts=E, max_rows=ROWS, top_k=K, hidden=H,
                             intermediate=I, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((R,), LIMIT, device="cuda"), limit=LIMIT,
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


def _run(cache, buffers, params, ids, x, weights, graph):
    from sglang.srt.layers.moe.gpu_expert_cache import cache_forward

    out = cache_forward(cache, hidden_states=x.cuda(), topk_ids=ids.cuda(),
                        topk_weights=weights.cuda(), buffers=buffers, params=params,
                        graph=graph)
    torch.cuda.synchronize()
    return out.cpu()


def _consistent(cache):
    slot_of, row_expert = cache.slot_of.cpu(), cache.row_expert.cpu()
    for row, e in enumerate(row_expert.tolist()):
        assert e < 0 or int(slot_of[e]) == row
    assert int((slot_of >= 0).sum()) == int((row_expert >= 0).sum())


@pytest.mark.parametrize("graph", [False, True])
def test_hits_admitted_and_overflow_misses_match_the_reference(setup, graph):
    from sglang.kernels.ops.moe.lk_verify_uva import STAT_ADMITTED, STAT_ROW_HITS

    experts, cache, buffers, params = setup
    # Rows 0..1 routed on every row, plus misses drawn from experts R..E-1.
    ids = torch.stack([torch.cat([torch.tensor([0, 1]), R + torch.randperm(E - R)[:K - 2]])
                       for _ in range(ROWS)]).int()
    ids[::2, -1] = -1  # EP-remote routes
    x = torch.randn(ROWS, H).bfloat16()
    weights = torch.rand(ROWS, K)
    expected = _reference(x, ids, weights, *experts, LIMIT)
    # From an empty cache too, where every route misses.
    for empty in (False, True):
        if empty:
            cache.slot_of.fill_(-1)
            cache.row_expert.fill_(-1)
        _close(_run(cache, buffers, params, ids, x, weights, graph), expected, cache.fp8)
        _consistent(cache)
    # Repeating the forward hits what the first one admitted.
    stats0 = cache.stats.clone()
    _close(_run(cache, buffers, params, ids, x, weights, graph), expected, cache.fp8)
    _consistent(cache)
    if graph:
        delta = (cache.stats - stats0).cpu()
        assert int(delta[STAT_ROW_HITS]) > 0
        assert int(delta[STAT_ADMITTED]) <= C
