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


@pytest.mark.parametrize("graph", [False, True])
def test_hits_and_misses_match_the_reference(graph):
    from sglang.srt.layers.moe.gpu_expert_cache import (
        ForwardBuffers,
        LayerCache,
        MoeParams,
        cache_forward,
    )

    torch.manual_seed(3)
    E, H, I, k, rows, S, limit = 12, 512, 256, 4, 9, 7, 10.0
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(119, 123, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(119, 123, (E, H, I // 32), dtype=torch.uint8)
    cache = LayerCache(layer_id=0, host_tensors=(w13, s13, w2, s2), num_slots=S, device="cuda")
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(rows)]).int()
    ids[::2, -1] = -1  # EP-remote routes
    x = torch.randn(rows, H).bfloat16()
    weights = torch.rand(rows, k)
    expected = _reference(x, ids, weights, w13, s13, w2, s2, limit)
    buffers = ForwardBuffers(num_experts=E, max_rows=rows, top_k=k, hidden=H,
                             intermediate=I, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((S,), limit, device="cuda"), limit=limit,
                       tp_size=1, tp_rank=0)

    def run():
        out = cache_forward(cache, hidden_states=x.cuda(), topk_ids=ids.cuda(),
                            topk_weights=weights.cuda(), buffers=buffers, params=params,
                            graph=graph)
        torch.cuda.synchronize()
        return out.cpu()

    # Cold start holds experts 0..S-1, so routes to S..E-1 take the UVA path.
    assert (ids >= S).any() and ((ids >= 0) & (ids < S)).any()
    mixed = run()
    torch.testing.assert_close(mixed, expected, rtol=3e-2, atol=3e-2 * expected.abs().max().item())
    # Unmapping every slot must not change the result: hit and miss paths agree.
    cache.slot_of.fill_(-1)
    all_miss = run()
    torch.testing.assert_close(all_miss, expected, rtol=3e-2, atol=3e-2 * expected.abs().max().item())
    assert torch.equal(cache.hist.cpu(), 2 * torch.bincount(
        ids.flatten()[ids.flatten() >= 0].long(), minlength=E).int())
