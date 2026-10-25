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


@pytest.fixture(scope="module")
def setup():
    """One cache for the module: registered host memory is never unregistered, so a
    second cache in this process could land on the same addresses (CUDA error 712)."""
    from sglang.srt.layers.moe.gpu_expert_cache import ForwardBuffers, LayerCache, MoeParams

    torch.manual_seed(3)
    experts = (
        torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8),
        torch.randint(119, 123, (E, 2 * I, H // 32), dtype=torch.uint8),
        torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8),
        torch.randint(119, 123, (E, H, I // 32), dtype=torch.uint8),
    )
    cache = LayerCache(layer_id=0, host_tensors=experts, num_slots=S, num_staging=P,
                       device="cuda")
    buffers = ForwardBuffers(num_experts=E, max_rows=ROWS, top_k=K, hidden=H,
                             intermediate=I, num_staging=P, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((S + P,), LIMIT, device="cuda"), limit=LIMIT,
                       tp_size=1, tp_rank=0)
    return experts, cache, buffers, params


@pytest.mark.parametrize("graph", [False, True])
def test_hits_staged_and_overflow_misses_match_the_reference(setup, graph):
    from sglang.srt.layers.moe.gpu_expert_cache import cache_forward

    experts, cache, buffers, params = setup
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(ROWS)]).int()
    ids[::2, -1] = -1  # EP-remote routes
    x = torch.randn(ROWS, H).bfloat16()
    weights = torch.rand(ROWS, K)
    expected = _reference(x, ids, weights, *experts, LIMIT)
    tol = dict(rtol=3e-2, atol=3e-2 * expected.abs().max().item())

    def run():
        hist0 = cache.hist.clone()
        out = cache_forward(cache, hidden_states=x.cuda(), topk_ids=ids.cuda(),
                            topk_weights=weights.cuda(), buffers=buffers, params=params,
                            graph=graph)
        torch.cuda.synchronize()
        local = ids.flatten()[ids.flatten() >= 0].long()
        assert torch.equal((cache.hist - hist0).cpu(), torch.bincount(local, minlength=E).int())
        return out.cpu()

    slot_of = cache.slot_of.clone()
    # Cold start holds experts 0..S-1; the other E-S > P experts are staged or overflow.
    assert len(set(ids[(ids >= S)].tolist())) > P
    torch.testing.assert_close(run(), expected, **tol)
    # With every slot unmapped all routes are misses: the paths must agree.
    cache.slot_of.fill_(-1)
    try:
        torch.testing.assert_close(run(), expected, **tol)
    finally:
        cache.slot_of.copy_(slot_of)
