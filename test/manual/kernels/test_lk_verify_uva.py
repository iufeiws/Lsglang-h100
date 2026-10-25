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


class _Cache:
    """One layer's residency state: expert <-> row tables, scores, counters."""

    def __init__(self, E, R, max_copy, device):
        from sglang.kernels.ops.moe.lk_verify_uva import MISS_BUCKETS, STAT_COUNT

        i32 = lambda *shape: torch.zeros(shape, dtype=torch.int32, device=device)  # noqa: E731
        self.slot_of = torch.full((E,), -1, dtype=torch.int32, device=device)
        self.slot_of[:R] = torch.arange(R, dtype=torch.int32, device=device)
        self.row_expert = torch.arange(R, dtype=torch.int32, device=device)
        self.score = torch.zeros(E, device=device)
        self.counts = i32(E)
        self.copy_row, self.copy_expert = i32(max_copy), i32(max_copy)
        self.stats, self.miss_hist = i32(STAT_COUNT), i32(MISS_BUCKETS)

    def admit(self, ids, decay):
        from sglang.kernels.ops.moe.lk_verify_uva import admit_misses

        slot_ids = torch.empty(ids.numel(), dtype=torch.int32, device=ids.device)
        miss_tab = torch.empty(ids.numel(), dtype=torch.int32, device=ids.device)
        admit_misses(topk_ids=ids, slot_of=self.slot_of, row_expert=self.row_expert,
                     score=self.score, counts=self.counts, copy_row=self.copy_row,
                     copy_expert=self.copy_expert, slot_ids=slot_ids, miss_tab=miss_tab,
                     stats=self.stats, miss_hist=self.miss_hist, decay=decay)
        return slot_ids.cpu(), miss_tab.cpu()


def test_admit_misses_evicts_unrouted_low_score_rows():
    """Missed experts (highest score first, at most the copy list) replace the experts
    of rows this forward does not route (lowest score first); the rest overflow to UVA,
    -1 routes are skipped, and the tables stay inverse of each other."""
    from sglang.kernels.ops.moe.lk_verify_uva import (
        STAT_ADMITTED,
        STAT_OVERFLOW,
        STAT_ROUTES,
        STAT_ROW_HITS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    E, R = 10, 4
    c = _Cache(E, R, max_copy=2, device=device)
    # Rows hold 0..3. Misses 5, 6, 7 tie at score 1; rows 2, 3 (experts 2, 3) are free.
    ids = torch.tensor([[0, 5, 6], [7, -1, 1]], dtype=torch.int32, device=device)
    slot_ids, miss_tab = c.admit(ids, decay=0.5)
    assert c.copy_row.cpu().tolist() == [2, 3] and c.copy_expert.cpu().tolist() == [5, 6]
    assert slot_ids.tolist() == [0, 2, 3, R, R, 1]
    assert miss_tab.tolist() == [0, 0, 0, -1, 0, 0]
    assert c.row_expert.cpu().tolist() == [0, 1, 5, 6]
    want = [-1] * E
    for row, e in enumerate(c.row_expert.cpu().tolist()):
        want[e] = row
    assert c.slot_of.cpu().tolist() == want
    st = c.stats.cpu()
    assert [int(st[i]) for i in (STAT_ROUTES, STAT_ROW_HITS, STAT_ADMITTED, STAT_OVERFLOW)] == [5, 2, 2, 1]
    assert c.miss_hist.cpu().tolist().index(1) == 3  # one forward, 3 distinct misses
    # Next forward: 7 (score 0.5 + 1) is admitted before 2 (0 + 1); every row is free,
    # all at score 0.5, so rows 0 and 1 go first.
    slot_ids, _ = c.admit(torch.tensor([[2, 7]], dtype=torch.int32, device=device), decay=0.5)
    assert c.copy_expert.cpu().tolist() == [7, 2] and c.copy_row.cpu().tolist() == [0, 1]
    assert slot_ids.tolist() == [1, 0]
    assert c.row_expert.cpu().tolist() == [7, 2, 5, 6]
    torch.testing.assert_close(c.score.cpu()[[0, 2, 5, 7]], torch.tensor([0.5, 1.0, 0.5, 1.5]))
    # A forward with no misses copies nothing.
    c.admit(torch.tensor([[5, 6]], dtype=torch.int32, device=device), decay=0.5)
    assert c.copy_expert.cpu().tolist() == [-1, -1]


def test_admitted_misses_are_copied_and_overflow_alone_runs_on_uva():
    """Admitted experts' blocks land in their rows (other rows untouched); only the
    overflow routes are computed by the UVA kernel."""
    from sglang.kernels.ops.moe.lk_verify_uva import copy_experts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(5)
    E, H, I, k, rows, R, C = 8, 128, 64, 3, 70, 4, 2
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)
    host, _storage = _scattered_host(w13, s13, w2, s2, torch.randperm(E), device)
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(rows)]).int()
    ids[::3, 1] = -1
    x = torch.randn(rows, H).bfloat16()
    weights = torch.rand(rows, k)
    to = lambda t: t.to(device)  # noqa: E731
    c = _Cache(E, R, max_copy=C, device=device)
    # Every expert is routed, so no row is free: nothing is admitted.
    slot_ids, miss_tab = c.admit(to(ids), decay=1.0)
    assert c.copy_expert.cpu().tolist() == [-1, -1]
    # Route only rows' experts 0 and 1 plus misses: rows 2, 3 are free for two admissions.
    ids = torch.stack([torch.tensor([r % 2, 4 + r % 4, -1]) for r in range(rows)]).int()
    slot_ids, miss_tab = c.admit(to(ids), decay=1.0)
    admitted = c.copy_expert.cpu().tolist()
    assert len(set(admitted)) == C and min(admitted) >= 4
    flat = ids.flatten()
    overflow = (flat >= 4) & ~torch.isin(flat, torch.tensor(admitted, dtype=torch.int32))
    assert torch.equal(miss_tab < 0, overflow)

    sources = [t.clone() for t in (w13, s13, w2, s2)]
    if device == "cuda":
        sources = [t.pin_memory() for t in sources]
    slabs = [torch.zeros((R, *t.shape[1:]), dtype=torch.uint8, device=device) for t in sources]
    i64 = lambda vals: torch.tensor(vals, dtype=torch.int64, device=device)  # noqa: E731
    words = [t[0].numel() // 4 for t in sources]
    copy_experts(copy_row=c.copy_row, copy_expert=c.copy_expert,
                 src_ptrs=i64([t.data_ptr() for t in sources]),
                 dst_ptrs=i64([slab.data_ptr() for slab in slabs]),
                 words=torch.tensor(words, dtype=torch.int32, device=device),
                 max_words=max(words), num_programs=3)
    for slab, src in zip(slabs, sources):
        for row in range(R):
            e = int(c.row_expert[row])
            if e in admitted:
                assert torch.equal(slab[row].cpu(), src[e].cpu())
            else:
                assert not slab[row].any()

    expected = _reference(x, ids, weights, miss_tab, w13, s13, w2, s2, 1, 0.0, True)
    scratch = UvaScratch.allocate(num_experts=E, max_slots=rows * k, intermediate=I, device=device)
    for eager_plan in (False, True):
        gout_part = torch.full((rows * k, H), float("nan"), device=device)
        launch_draft_miss_uva(
            hidden_states=to(x), topk_ids=to(ids), topk_weights=to(weights),
            slot_tab=to(miss_tab), gout_part=gout_part, rows_per_request=1,
            host=host, scratch=scratch, activation=0, alpha=1.702, limit=0.0,
            include_first_row=True, eager_plan=eager_plan,
        )
        assert torch.isnan(gout_part.cpu()[~overflow]).all()
        torch.testing.assert_close(gout_part.cpu()[overflow], expected[overflow], rtol=2e-4, atol=1e-3)
