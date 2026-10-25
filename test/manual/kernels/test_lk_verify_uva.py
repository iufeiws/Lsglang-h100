"""Draft-row pool misses computed over UVA must equal a dense MXFP4 reference.

Runs on CUDA (host tensors pinned) or on CPU with TRITON_INTERPRET=1:
  TRITON_INTERPRET=1 python -m pytest -q test/manual/kernels/test_lk_verify_uva.py
"""

import pytest
import torch

from sglang.kernels.ops.moe.lk_verify_uva import (
    HostExperts,
    UvaScratch,
    launch_draft_miss_uva,
    matches_host_experts,
)

_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _dequant(packed, scales):
    """[..., N, K/2] uint8 + [..., N, K/32] e8m0 -> [..., N, K] fp32, low nibble first."""
    lo = _E2M1[(packed & 0xF).long()]
    hi = _E2M1[(packed >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(-2)
    scale = torch.pow(2.0, scales.float() - 127).repeat_interleave(32, dim=-1)
    return w * scale


def _reference(x, ids, weights, slot_tab, w13, s13, w2, s2, rows_per_request, limit, include_first):
    rows, k = ids.shape
    H = x.shape[1]
    I = w13.shape[1] // 2
    out = torch.zeros(rows * k, H)
    for slot in range(rows * k):
        row, e = slot // k, int(ids.flatten()[slot])
        if e < 0 or slot_tab[slot] >= 0 or (row % rows_per_request == 0 and not include_first):
            continue
        w_up = _dequant(w13[e], s13[e])
        h = w_up @ x[row].float()
        gate, up = h[:I], h[I:]
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        act = up * gate * torch.sigmoid(gate)
        out[slot] = weights.flatten()[slot] * (_dequant(w2[e], s2[e]) @ act)
    return out


def _scattered_host(w13, s13, w2, s2, placement, device):
    """Store expert e at position placement[e] of separate buffers, as a backend might."""
    storage, tables = [], []
    for t in (w13, s13, w2, s2):
        buf = torch.empty_like(t)
        buf[placement] = t
        if device == "cuda":
            buf = buf.pin_memory()
        step = buf[0].numel() * buf.element_size()
        storage.append(buf)
        tables.append([buf.data_ptr() + int(placement[e]) * step for e in range(t.shape[0])])
    host = HostExperts.from_addresses(
        w13=tables[0], s13=tables[1], w2=tables[2], s2=tables[3],
        hidden=w13.shape[2] * 2, intermediate=w13.shape[1] // 2, device=device)
    return host, storage


def test_self_check_rejects_wrong_addresses():
    """A provider whose addresses do not hold sglang's expert bytes must be refused."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E, H, I = 4, 64, 64
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)
    refs = dict(w13=w13, s13=s13, w2=w2, s2=s2, experts=range(E))
    host, _storage = _scattered_host(w13, s13, w2, s2, torch.tensor([2, 0, 3, 1]), device)
    assert matches_host_experts(host, **refs)
    swapped = HostExperts(
        w13_ptrs=host.w13_ptrs.flip(0), s13_ptrs=host.s13_ptrs, w2_ptrs=host.w2_ptrs,
        s2_ptrs=host.s2_ptrs, num_experts=E, hidden=H, intermediate=I)
    assert not matches_host_experts(swapped, **refs)


@pytest.mark.parametrize("requests", [1, 3])
# 0.5 is inside the gate range, so the one-sided gate clamp is exercised.
@pytest.mark.parametrize("limit", [0.0, 0.5])
@pytest.mark.parametrize("include_first", [False, True])
def test_draft_miss_matches_reference(requests, limit, include_first):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(requests * 10 + int(limit))
    E, H, I, k, rpr = 12, 256, 128, 4, 3
    rows = requests * rpr
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)
    host, _storage = _scattered_host(w13, s13, w2, s2, torch.randperm(E), device)
    # Distinct experts per row; EP-remote routes are -1; one expert shared by all rows.
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(rows)]).int()
    ids[:, 0] = 5
    ids[::2, -1] = -1
    slot_tab = torch.where(torch.rand(rows * k) < 0.4, torch.randint(0, 8, (rows * k,)), -1).int()
    x = torch.randn(rows, H).bfloat16()
    weights = torch.rand(rows, k)
    expected = _reference(x, ids, weights, slot_tab, w13, s13, w2, s2, rpr, limit, include_first)

    to = lambda t: t.to(device)  # noqa: E731
    scratch = UvaScratch.allocate(num_experts=E, max_slots=rows * k, intermediate=I, device=device)
    gout_part = torch.full((rows * k, H), float("nan"), device=device)
    keep = to(torch.tensor([
        int(ids.flatten()[s]) < 0 or slot_tab[s] >= 0
        or ((s // k) % rpr == 0 and not include_first)
        for s in range(rows * k)
    ]))
    launch_draft_miss_uva(
        hidden_states=to(x), topk_ids=to(ids), topk_weights=to(weights),
        slot_tab=to(slot_tab), gout_part=gout_part, rows_per_request=rpr,
        host=host, scratch=scratch, activation=0, alpha=1.702, limit=limit,
        include_first_row=include_first,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    # Slots the kernel does not own stay untouched (the pool or the CPU fills them).
    assert torch.isnan(gout_part[keep]).all()
    computed = ~keep.cpu()
    assert computed.any()
    torch.testing.assert_close(gout_part.cpu()[computed], expected[computed], rtol=2e-4, atol=1e-3)


def test_cache_mode_computes_every_route_and_counts_them():
    """Without a pool slot table every local route is computed and counted in ``hist``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    E, H, I, k, rows = 10, 128, 64, 3, 4
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)
    host, _storage = _scattered_host(w13, s13, w2, s2, torch.randperm(E), device)
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(rows)]).int()
    ids[1, 0] = -1
    x = torch.randn(rows, H).bfloat16()
    weights = torch.rand(rows, k)
    no_pool = torch.full((rows * k,), -1, dtype=torch.int32)
    expected = _reference(x, ids, weights, no_pool, w13, s13, w2, s2, 1, 0.0, True)

    to = lambda t: t.to(device)  # noqa: E731
    scratch = UvaScratch.allocate(num_experts=E, max_slots=rows * k, intermediate=I, device=device)
    hist = torch.zeros(E, dtype=torch.int32, device=device)
    for _ in range(2):
        gout_part = torch.zeros((rows * k, H), device=device)
        launch_draft_miss_uva(
            hidden_states=to(x), topk_ids=to(ids), topk_weights=to(weights),
            slot_tab=None, gout_part=gout_part, rows_per_request=1,
            host=host, scratch=scratch, activation=0, alpha=1.702, limit=0.0,
            include_first_row=True, hist=hist,
        )
    torch.testing.assert_close(gout_part.cpu(), expected, rtol=2e-4, atol=1e-3)
    local = ids.flatten()[ids.flatten() >= 0].long()
    assert torch.equal(hist.cpu(), 2 * torch.bincount(local, minlength=E).int())


def test_eager_plan_splits_hot_experts_across_tiles():
    """Host-planned batches past one tile per expert (prefill tails) must cover every route."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)
    E, H, I, k, rows = 6, 128, 64, 2, 150
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)
    host, _storage = _scattered_host(w13, s13, w2, s2, torch.randperm(E), device)
    # Expert 2 takes every row (3 tiles of 64); expert 4 is never routed.
    ids = torch.stack([torch.tensor([2, int(torch.tensor([0, 1, 3, 5])[r % 4])]) for r in range(rows)]).int()
    ids[7, 1] = -1
    x = torch.randn(rows, H).bfloat16()
    weights = torch.rand(rows, k)
    no_pool = torch.full((rows * k,), -1, dtype=torch.int32)
    expected = _reference(x, ids, weights, no_pool, w13, s13, w2, s2, 1, 0.0, True)

    to = lambda t: t.to(device)  # noqa: E731
    scratch = UvaScratch.allocate(num_experts=E, max_slots=1, intermediate=I, device=device)
    hist = torch.zeros(E, dtype=torch.int32, device=device)
    gout_part = torch.zeros((rows * k, H), device=device)
    launch_draft_miss_uva(
        hidden_states=to(x), topk_ids=to(ids), topk_weights=to(weights),
        slot_tab=None, gout_part=gout_part, rows_per_request=1,
        host=host, scratch=scratch, activation=0, alpha=1.702, limit=0.0,
        include_first_row=True, hist=hist, eager_plan=True,
    )
    torch.testing.assert_close(gout_part.cpu(), expected, rtol=2e-4, atol=1e-3)
    local = ids.flatten()[ids.flatten() >= 0].long()
    assert torch.equal(hist.cpu(), torch.bincount(local, minlength=E).int())
