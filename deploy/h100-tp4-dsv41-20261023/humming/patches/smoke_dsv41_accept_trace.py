"""CPU smoke test for the gated accept-trace probe.  ``[accept-trace]``

Runs the real ``AcceptTrace`` class (the one installed into site-packages) on
CPU tensors, with ``torch.cuda.synchronize`` stubbed out, and checks:

1. gate OFF  -> ``maybe_observe`` is a no-op (no file, no buffer);
2. gate ON   -> per-step / per-request / per-slot rows reach the jsonl;
3. the recorded ``glen`` / ``tk_ids`` / ``dval`` / ``drank`` match a hand
   computed expectation, including a deliberate mismatch pattern
   (glen = 5 / 2 / 0 for the three synthetic requests);
4. changing ``bs`` between steps inside one window is handled.

Usage:  python smoke_dsv41_accept_trace.py [outdir]
"""

from __future__ import annotations

import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/SSD01/models/.llm-env/Lsglang/logs"
OUT = os.path.join(OUTDIR, "smoke-acctrace.jsonl")
if os.path.exists(OUT):
    os.remove(OUT)

import torch  # noqa: E402

torch.cuda.synchronize = lambda *a, **k: None  # CPU-only run


def build(bs, nslots=6, V=64):
    """Build a synthetic verify batch with a known greedy-match profile."""
    g = torch.Generator().manual_seed(7 + bs)
    tgt = torch.randint(0, V, (bs, nslots), generator=g)
    cand = torch.zeros(bs, nslots, dtype=torch.int64)
    glen = torch.zeros(bs, dtype=torch.int32)
    for i in range(bs):
        cand[i, 0] = 900 + i  # current token (root)
        gl = i % (nslots)  # 0,1,2,... mismatch profile
        if bs == 3:
            gl = [5, 2, 0][i]
        glen[i] = gl
        for s in range(1, nslots):
            cand[i, s] = tgt[i, s - 1] if s <= gl else (tgt[i, s - 1] + 1) % V
    logits = torch.full((bs * nslots, V), -5.0)
    logits.scatter_(1, tgt.view(-1, 1), 3.0)
    # rank of the (rejected) draft token: make its logit a distinct, known value
    for i in range(bs):
        for s in range(1, nslots):
            if s > int(glen[i]):
                logits[i * nslots + (s - 1), cand[i, s]] = 1.0 + 0.01 * s
    acc = glen.clone()
    commit = acc + 1
    trim = torch.zeros(bs, dtype=torch.int32)
    bonus = tgt[torch.arange(bs), acc.to(torch.int64)]
    return logits, cand.view(bs, nslots), glen, acc, commit, trim, bonus


def run(enabled: bool, tag: str):
    for k in (
        "SGLANG_DSV41_PROBE_ACCEPT_TRACE",
        "SGLANG_DSV41_PROBE_ACCEPT_EVERY",
        "SGLANG_DSV41_PROBE_ACCEPT_TOPK",
        "SGLANG_DSV41_PROBE_ACCEPT_TOPK_STEPS",
        "SGLANG_DSV41_PROBE_ACCEPT_OUT",
    ):
        os.environ.pop(k, None)
    if enabled:
        os.environ["SGLANG_DSV41_PROBE_ACCEPT_TRACE"] = "1"
    os.environ["SGLANG_DSV41_PROBE_ACCEPT_EVERY"] = "2"
    os.environ["SGLANG_DSV41_PROBE_ACCEPT_TOPK"] = "4"
    os.environ["SGLANG_DSV41_PROBE_ACCEPT_TOPK_STEPS"] = "8"
    path = OUT.replace(".jsonl", f".{tag}.jsonl")
    if os.path.exists(path):
        os.remove(path)
    os.environ["SGLANG_DSV41_PROBE_ACCEPT_OUT"] = path
    for m in list(sys.modules):
        if "dsv41_accept_trace" in m:
            del sys.modules[m]
    from sglang.srt.utils.dsv41_accept_trace import dsv41_accept_trace as probe

    assert probe.enabled is enabled, probe.enabled
    steps = [3, 1, 2, 3]  # bs changes inside/among windows
    expect = {}
    for k, bs in enumerate(steps):
        logits, cand, glen, acc, commit, trim, bonus = build(bs)
        probe.maybe_observe(
            tp_rank=0,
            forward_ct=100 + k,
            rids=[f"r{k}-{i}" for i in range(bs)],
            bs=bs,
            verify_ids_2d=cand,
            target_logits=logits,
            draft_block=None,
            sampling_info=None,
            correct_len=acc,
            cap_trim_lens=trim,
            bonus=bonus,
            commit_lens=commit,
            req_pool_indices=torch.arange(bs, dtype=torch.int32) + 10,
            layout=None,
        )
        expect[k] = (bs, glen.tolist(), cand.tolist())
    probe._atexit()
    path = os.environ["SGLANG_DSV41_PROBE_ACCEPT_OUT"]
    if not enabled:
        assert not os.path.exists(path), f"gate OFF wrote a file: {path}"
        print("OK gate OFF: no file, no buffers")
        return
    recs = {}
    lines = [json.loads(x) for x in open(path) if x.strip()]
    kinds = [x["kind"] for x in lines]
    assert kinds.count("flush") >= 2 and kinds[-1] == "summary", kinds
    for ln in lines:
        if ln["kind"] != "flush":
            continue
        for st in ln["steps"]:
            assert len(st["records"]) == st["bs"]
            recs[st["step"]] = st
    assert sorted(recs) == [100, 101, 102, 103], sorted(recs)
    n_tot = 0
    for k in range(4):
        bs, gl, cd = expect[k]
        st = recs[100 + k]
        assert st["bs"] == bs, (k, st["bs"], bs)
        n_tot += bs
        for i, r in enumerate(st["records"]):
            assert r["glen"] == gl[i], (k, i, r["glen"], gl[i])
            assert r["cand"] == cd[i], (k, i, r["cand"], cd[i])
            assert r["acc"] == gl[i] and r["commit"] == gl[i] + 1
            assert r["ttmp"] == 0.0 and r["dtmp"] == 0.0
            assert r["tk_row"] == min(gl[i], 4)
            if gl[i] < 5:
                # dumped row is the row of the first mismatched slot
                row = gl[i]
                assert r["drank"] == 1, (r["drank"], row)  # only the argmax outranks
                assert abs(r["tk_vals"][0] - 3.0) < 1e-5, r["tk_vals"][0]
                assert abs(r["dval"] - (1.0 + 0.01 * (1 + row))) < 1e-5, (r["dval"], row)
    counts = [ln for ln in lines if ln["kind"] == "flush"][-1]["counters"]
    print("OK gate ON:", json.dumps(
        {"records": n_tot, "glen_eq_acc": counts["glen_eq_acc"],
         "glen_eq_acc_tot": counts["glen_eq_acc_tot"],
         "bonus_ok": counts["bonus_ok"], "bonus_tot": counts["bonus_tot"]}, ))
    summ = [ln for ln in lines if ln["kind"] == "summary"][0]
    assert summ["cumulative"].get("skip_shape", 0) == 0, summ["cumulative"]
    print("cumulative:", json.dumps(summ["cumulative"]))
    assert summ["cumulative"].get("errors", 0) == 0, summ["cumulative"]
    assert summ["cumulative"]["steps"] == 4
    assert summ["cumulative"]["topk_steps_done"] == 4
    assert summ["cumulative"]["records"] == n_tot
    print(f"jsonl: {path}")


if __name__ == "__main__":
    run(False, "off")
    run(True, "on")
    print("ALL OK")
