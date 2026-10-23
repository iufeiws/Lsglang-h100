#!/usr/bin/env python3
"""Build the staged (patched) copies of the 5 sglang files for the DSV4.1
segment probe, from the pristine repo tree.  Idempotent, anchor-checked.

usage: stage_probe_patch.py <src_pkg_root> <dst_pkg_root>
  <src_pkg_root>  e.g. /mnt/SSD01/models/.llm-env/Lsglang/lsglang/python
  <dst_pkg_root>  staging dir; writes <dst>/sglang/...
"""
from __future__ import annotations

import os
import shutil
import sys

IMPORT_BLOCK = '''
# [seg-probe] gated segment timing; default OFF, no behaviour change when off.
# Gate: SGLANG_DSV41_PROBE_SEGMENTS=1  (see docs/probe-verify-step-segments.md)
try:
    from sglang.srt.utils.dsv41_segment_probe import (
        call as _seg_probe_call,
        ensure_channels as _seg_probe_ensure_channels,
        on_step as _seg_probe_on_step,
        span as _seg_probe_span,
    )
except Exception:  # pragma: no cover
    from contextlib import nullcontext as _seg_probe_nullcontext

    def _seg_probe_span(*_a, **_k):
        return _seg_probe_nullcontext()

    def _seg_probe_call(_name, _fn, *_a, **_k):
        return _fn(*_a, **_k)

    def _seg_probe_on_step():
        return None

    def _seg_probe_ensure_channels():
        return None
'''

SPAN_ONLY_BLOCK = '''
# [seg-probe] gated segment timing; default OFF, no behaviour change when off.
try:
    from sglang.srt.utils.dsv41_segment_probe import span as _seg_probe_span
except Exception:  # pragma: no cover
    from contextlib import nullcontext as _seg_probe_nullcontext

    def _seg_probe_span(*_a, **_k):
        return _seg_probe_nullcontext()
'''


#: single-line anchor at the end of each file's import section, after which the
#: probe import block is inserted.
IMPORT_ANCHOR = {
    "sglang/srt/layers/moe/fused_moe_triton/layer.py": "_LK_POOL_PROBE_N = 0\n",
    "sglang/srt/speculative/dspark_components/dspark_draft.py": (
        "from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite\n"
    ),
    "sglang/kernels/ops/speculative/dspark/markov_sample.py": (
        "from sglang.srt.utils import is_npu\n"
    ),
    "sglang/kernels/ops/speculative/dspark/dspark_accept.py": "_is_npu = is_npu()\n",
    "sglang/srt/speculative/dspark_components/dspark_worker_v2.py": (
        "from sglang.srt.server_args import ServerArgs\n"
    ),
}

IMPORT_BLOCK_FULL_FILES = {
    "sglang/srt/layers/moe/fused_moe_triton/layer.py",
    "sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
}


def patch(path: str, rel: str, edits) -> str:
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    out = src
    imports = IMPORT_BLOCK if rel in IMPORT_BLOCK_FULL_FILES else SPAN_ONLY_BLOCK
    anchor = IMPORT_ANCHOR[rel]
    if anchor not in out:
        raise SystemExit(f"IMPORT ANCHOR missing in {path}: {anchor!r}")
    out = out.replace(anchor, anchor + imports, 1)
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"ANCHOR {n}!=1 in {path}: {old[:90]!r}")
        out = out.replace(old, new)
    return out


EDITS = {}

# ---------------------------------------------------------------- 1. layer.py
EDITS["sglang/srt/layers/moe/fused_moe_triton/layer.py"] = [
    (
        "    def _init_pool(self):\n"
        "        if not (self.is_pool_layer or self.is_gpu_prefill_layer):\n"
        "            return\n",
        "    def _init_pool(self):\n"
        "        # [seg-probe] allocate the tick rings eagerly (never inside a capture)\n"
        "        _seg_probe_ensure_channels()\n"
        "        if not (self.is_pool_layer or self.is_gpu_prefill_layer):\n"
        "            return\n",
    ),
    (
        "    def _cpu_decode(self, hidden_states, topk_weights, topk_ids):\n"
        "        bs = hidden_states.size(0)\n",
        "    def _cpu_decode(self, hidden_states, topk_weights, topk_ids):\n"
        "        # [seg-probe] bracket one pool layer's whole pool path.\n"
        "        # Covers _pool_decode_launch + lk_moe.cpu_decode + wait_stream +\n"
        "        # output_gpu.add_ + the .to(dtype) copy; being main-stream bracketed\n"
        "        # it CONTAINS the cross-stream wait for the pool stream.\n"
        "        with _seg_probe_span(\"pool_layer\", idx=self.layer_id):\n"
        "            return self._cpu_decode_impl(hidden_states, topk_weights, topk_ids)\n"
        "\n"
        "    def _cpu_decode_impl(self, hidden_states, topk_weights, topk_ids):\n"
        "        bs = hidden_states.size(0)\n",
    ),
]

# ---------------------------------------------------------- 2. dspark_draft.py
LOOP_OLD = """    for step in range(gamma):
        prev_tokens, state = _draft_step(
            markov_head,
            prev_tokens,
            base_logits[:, step],
            hidden_states=None if draft_hidden is None else draft_hidden[:, step],
            state=state,
            temperatures=step_temperatures,
            noise=noise,
            step=step,
            logits_out=None if greedy_draft else logits_out[:, step],
        )
        prev_tokens = tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev_tokens)
        draft_tokens.append(prev_tokens)
"""
LOOP_NEW = """    with _seg_probe_span("draft5_total"):
        for step in range(gamma):
            with _seg_probe_span("draft_step", idx=step):
                prev_tokens, state = _draft_step(
                    markov_head,
                    prev_tokens,
                    base_logits[:, step],
                    hidden_states=None if draft_hidden is None else draft_hidden[:, step],
                    state=state,
                    temperatures=step_temperatures,
                    noise=noise,
                    step=step,
                    logits_out=None if greedy_draft else logits_out[:, step],
                )
            with _seg_probe_span("draft_bcast", idx=step):
                prev_tokens = tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev_tokens)
            draft_tokens.append(prev_tokens)
"""
EDITS["sglang/srt/speculative/dspark_components/dspark_draft.py"] = [
    (LOOP_OLD, LOOP_NEW),
]

# ------------------------------------------------------- 3. markov_sample.py
EDITS["sglang/kernels/ops/speculative/dspark/markov_sample.py"] = [
    (
        "            full = gather_logits(local)[:, : logits_out.shape[1]]\n",
        "            with _seg_probe_span(\"vocab_ag\", idx=step):\n"
        "                full = gather_logits(local)[:, : logits_out.shape[1]]\n",
    ),
    (
        "            gathered = partial if gather is None else gather(partial)\n",
        "            if gather is None:\n"
        "                gathered = partial\n"
        "            else:\n"
        "                with _seg_probe_span(\"vocab_ag\", idx=step):\n"
        "                    gathered = gather(partial)\n",
    ),
]

# ------------------------------------------------------- 4. dspark_accept.py
SAMPLING_OLD = """    sampling_fn(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
"""
SAMPLING_NEW = """    with _seg_probe_span("accept_kernel"):
        sampling_fn(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
"""
EDITS["sglang/kernels/ops/speculative/dspark/dspark_accept.py"] = [
    (SAMPLING_OLD, SAMPLING_NEW),
]

# ------------------------------------------------------- 5. dspark_worker_v2.py
ACCEPT_OLD = """        accept = self._verify_executor.accept_and_finalize(
"""
ACCEPT_NEW = """        accept = _seg_probe_call(
            "accept_all", self._verify_executor.accept_and_finalize,
"""
EDITS["sglang/srt/speculative/dspark_components/dspark_worker_v2.py"] = [
    (
        "    ) -> GenerationBatchResult:\n        if not self._hosts_draft:\n",
        "    ) -> GenerationBatchResult:\n"
        "        _seg_probe_on_step()  # [seg-probe] one call per decode step\n"
        "        if not self._hosts_draft:\n",
    ),
    (
        "        return self._forward_decode(batch, on_publish, grammar_barrier)\n"
        "\n    def _forward_dp_spec_prefill_coordination(\n",
        "        # [seg-probe] whole-iteration bracket (eager events, per step)\n"
        "        with _seg_probe_span(\"step_total\"):\n"
        "            return self._forward_decode(batch, on_publish, grammar_barrier)\n"
        "\n    def _forward_dp_spec_prefill_coordination(\n",
    ),
    (
        "        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):\n",
        "        with self._draft_context(), self._observers.segment(\n"
        "            InfoSegment.DRAFT\n"
        "        ), _seg_probe_span(\"draft_propose\"):\n",
    ),
    (
        "        with self._observers.segment(InfoSegment.TARGET_VERIFY):\n",
        "        with self._observers.segment(\n"
        "            InfoSegment.TARGET_VERIFY\n"
        "        ), _seg_probe_span(\"target_verify\"):\n",
    ),
    (ACCEPT_OLD, ACCEPT_NEW),
]


def main() -> None:
    src_root, dst_root = sys.argv[1], sys.argv[2]
    for rel, edits in EDITS.items():
        src = os.path.join(src_root, rel)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        out = patch(src, rel, edits)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(out)
        print("staged", rel)
    # the new probe module itself
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(dst_root, "sglang/srt/utils/dsv41_segment_probe.py")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(os.path.join(here, "dsv41_segment_probe.py"), dst)
    print("staged sglang/srt/utils/dsv41_segment_probe.py")


if __name__ == "__main__":
    main()
