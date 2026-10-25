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


def _assign(ids, slot_of, stage_expert, *, S, P, route=True, stats=None, miss_hist=None,
            hist=None):
    from sglang.kernels.ops.moe.lk_verify_uva import MISS_BUCKETS, STAT_COUNT, assign_slots

    device = ids.device
    E = slot_of.numel()
    out = dict(
        slot_ids=torch.empty(ids.numel(), dtype=torch.int32, device=device),
        miss_tab=torch.empty(ids.numel(), dtype=torch.int32, device=device),
        copy_expert=torch.empty(P, dtype=torch.int32, device=device),
        stats=stats if stats is not None else torch.zeros(STAT_COUNT, dtype=torch.int32, device=device),
        miss_hist=miss_hist if miss_hist is not None
        else torch.zeros(MISS_BUCKETS, dtype=torch.int32, device=device),
        hist=hist if hist is not None else torch.zeros(E, dtype=torch.int32, device=device),
    )
    assign_slots(topk_ids=ids, slot_of=slot_of,
                 mark=torch.empty(E, dtype=torch.int32, device=device),
                 free_rows=torch.empty(P, dtype=torch.int32, device=device),
                 stage_expert=stage_expert, copy_expert=out["copy_expert"],
                 slot_ids=out["slot_ids"], miss_tab=out["miss_tab"], hist=out["hist"],
                 stats=out["stats"], miss_hist=out["miss_hist"], num_slots=S,
                 num_staging=P, route=route)
    return {k: v.cpu() for k, v in out.items()}


def test_assign_slots_keeps_held_rows_and_prefetches():
    """Missed experts already in a staging row (prefetched or left by an earlier forward)
    are served without a copy, others take rows not holding a missed expert, and the
    remainder overflow to UVA; a prefetch only fills rows."""
    from sglang.kernels.ops.moe.lk_verify_uva import (
        STAT_OVERFLOW,
        STAT_PREFETCHED,
        STAT_ROUTES,
        STAT_SLOT_HITS,
        STAT_STAGE_COPIED,
        STAT_STAGE_KEPT,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    E, S, P = 10, 3, 3
    slot_of = torch.full((E,), -1, dtype=torch.int32)
    slot_of[torch.tensor([0, 1, 2])] = torch.arange(S, dtype=torch.int32)
    slot_of = slot_of.to(device)
    stage_expert = torch.full((P,), -1, dtype=torch.int32, device=device)
    # Prefetch predicts 7 (a miss) and 1 (in a slot, ignored): 7 takes row 0.
    pf = _assign(torch.tensor([[7, 1]], dtype=torch.int32, device=device), slot_of,
                 stage_expert, S=S, P=P, route=False)
    assert pf["copy_expert"].tolist() == [7, -1, -1]
    assert int(pf["stats"][STAT_PREFETCHED]) == 1
    assert stage_expert.cpu().tolist() == [7, -1, -1]
    # The forward misses 4, 5, 6, 7 (and 9 twice); remote -1 is skipped.
    ids = torch.tensor([[0, 7, 4], [5, -1, 9], [6, 9, 2]], dtype=torch.int32, device=device)
    out = _assign(ids, slot_of, stage_expert, S=S, P=P)
    # 7 keeps row 0; 4 and 5 take rows 1 and 2 (expert order); 6 and 9 overflow.
    assert out["copy_expert"].tolist() == [-1, 4, 5]
    assert stage_expert.cpu().tolist() == [7, 4, 5]
    skip = S + P
    assert out["slot_ids"].tolist() == [0, S + 0, S + 1, S + 2, skip, skip, skip, skip, 2]
    assert out["miss_tab"].tolist() == [0, 0, 0, 0, 0, -1, -1, -1, 0]
    st = out["stats"]
    assert (int(st[STAT_ROUTES]), int(st[STAT_SLOT_HITS])) == (8, 2)
    assert (int(st[STAT_STAGE_KEPT]), int(st[STAT_STAGE_COPIED]), int(st[STAT_OVERFLOW])) == (1, 2, 2)
    assert out["miss_hist"].tolist().index(1) == 5  # one forward with 5 distinct misses
    assert out["hist"].tolist() == [1, 0, 1, 0, 1, 1, 1, 1, 0, 2]
    # Next forward misses only 5 and 9: 5 keeps row 2, 9 refills a row held by 7 or 4.
    out = _assign(torch.tensor([[5, 9]], dtype=torch.int32, device=device), slot_of,
                  stage_expert, S=S, P=P)
    assert out["copy_expert"].tolist() == [9, -1, -1]
    assert out["slot_ids"].tolist() == [S + 2, S + 0]
    assert stage_expert.cpu().tolist() == [9, 4, 5]


def test_assign_slots_stages_misses_and_overflows_the_rest_to_uva():
    """Staged misses are copied into their rows with few persistent programs; overflow
    routes alone are computed by the UVA kernel."""
    from sglang.kernels.ops.moe.lk_verify_uva import stage_copy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(5)
    E, H, I, k, rows, S, P = 8, 128, 64, 3, 70, 3, 2
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)
    host, _storage = _scattered_host(w13, s13, w2, s2, torch.randperm(E), device)
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(rows)]).int()
    ids[::3, 1] = -1
    slot_of = torch.full((E,), -1, dtype=torch.int32)
    slot_of[torch.tensor([6, 1, 0])] = torch.arange(S, dtype=torch.int32)
    x = torch.randn(rows, H).bfloat16()
    weights = torch.rand(rows, k)

    to = lambda t: t.to(device)  # noqa: E731
    stage_expert = torch.full((P,), -1, dtype=torch.int32, device=device)
    out = _assign(to(ids), to(slot_of), stage_expert, S=S, P=P)
    flat = ids.flatten().long()
    local = flat >= 0
    slot = torch.where(local, slot_of[flat.clamp(min=0)], -1)
    missed = sorted(set(flat[local & (slot < 0)].tolist()))
    assert len(missed) > P
    assert out["copy_expert"].tolist() == missed[:P]
    overflow = local & (slot < 0) & (out["slot_ids"] == S + P)
    assert torch.equal(out["miss_tab"] < 0, overflow)

    # The copy reads each staged expert's 4 blocks and leaves the other rows alone.
    sources = [t.clone() for t in (w13, s13, w2, s2)]
    if device == "cuda":
        sources = [t.pin_memory() for t in sources]
    slabs = [torch.zeros((S + P, *t.shape[1:]), dtype=torch.uint8, device=device) for t in sources]
    i64 = lambda vals: torch.tensor(vals, dtype=torch.int64, device=device)  # noqa: E731
    words = [t[0].numel() // 4 for t in sources]
    stage_copy(copy_expert=to(out["copy_expert"]),
               src_ptrs=i64([t.data_ptr() for t in sources]),
               dst_ptrs=i64([slab[S].data_ptr() for slab in slabs]),
               words=torch.tensor(words, dtype=torch.int32, device=device),
               num_staging=P, max_words=max(words), num_programs=3)
    for slab, src in zip(slabs, sources):
        assert not slab[:S].any()
        for j, e in enumerate(missed[:P]):
            assert torch.equal(slab[S + j].cpu(), src[e].cpu())

    miss_tab = to(out["miss_tab"])
    expected = _reference(x, ids, weights, out["miss_tab"], w13, s13, w2, s2, 1, 0.0, True)
    scratch = UvaScratch.allocate(num_experts=E, max_slots=rows * k, intermediate=I, device=device)
    for eager_plan in (False, True):
        gout_part = torch.full((rows * k, H), float("nan"), device=device)
        launch_draft_miss_uva(
            hidden_states=to(x), topk_ids=to(ids), topk_weights=to(weights),
            slot_tab=miss_tab, gout_part=gout_part, rows_per_request=1,
            host=host, scratch=scratch, activation=0, alpha=1.702, limit=0.0,
            include_first_row=True, eager_plan=eager_plan,
        )
        assert torch.isnan(gout_part.cpu()[~overflow]).all()
        torch.testing.assert_close(gout_part.cpu()[overflow], expected[overflow], rtol=2e-4, atol=1e-3)
