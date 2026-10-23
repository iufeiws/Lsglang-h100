#!/usr/bin/env python
"""最小独立复现：DSPARK 拒绝采样路径里的 q>1 容差缺陷（不加载模型/专家权重；单卡）。

A) 用 DSPARK 真正的 softmax 路径（softmax_temp_flashinfer → flashinfer.sampling.softmax，
   fp32 logits + 每行温度）看本机是否会产出 q > 1.0
B) 当 q 略微 > 1.0 时，chain_speculative_sampling_triton 的接受长度是否被错误清零

安装版源码依据：
  sglang/kernels/ops/speculative/dspark/dspark_accept.py
    :413 softmax_temp_flashinfer → _flashinfer_softmax(logits=logits_2d, temperature=temp_per_row)
    :140 sampling_fn = chain_speculative_sampling_triton   (threshold_single/acc = 1.0)
  sglang/kernels/ops/speculative/reject_sampling.py
    :148  q_is_prob = (q > 0.0) & (q <= 1.0)      # 普通验证
    :149  if q_is_prob & (coin * q < p):
    :71   ... 同类判断（block 验证）    :194/:222  q_val 的同类 clamp（两处 residual）
"""
import inspect
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import softmax_temp_flashinfer
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)


def part_a(dev):
    print("=" * 72)
    print("A) DSPARK 的 softmax 路径是否产出 q > 1.0")
    try:
        from flashinfer.sampling import softmax as fi
        print("   flashinfer.sampling.softmax 签名:", inspect.signature(fi))
    except Exception as e:  # noqa: BLE001
        print("   !! 不可用:", repr(e))
        return None
    torch.manual_seed(0)
    ROWS, V = 6, 8192
    cases = {
        "N(0,1)": lambda: torch.randn(ROWS, V, device=dev),
        "N(0,5)": lambda: torch.randn(ROWS, V, device=dev) * 5,
        "N(0,20)": lambda: torch.randn(ROWS, V, device=dev) * 20,
        "N(0,50)": lambda: torch.randn(ROWS, V, device=dev) * 50,
        "N(0,100)": lambda: torch.randn(ROWS, V, device=dev) * 100,
        "peaked(+30)": lambda: torch.zeros(ROWS, V, device=dev).scatter_(
            1, torch.randint(0, V, (ROWS, 1), device=dev), 30.0),
        "peaked(+100)": lambda: torch.full((ROWS, V), -100.0, device=dev).scatter_(
            1, torch.randint(0, V, (ROWS, 1), device=dev), 100.0),
        "peaked(+300)": lambda: torch.full((ROWS, V), -300.0, device=dev).scatter_(
            1, torch.randint(0, V, (ROWS, 1), device=dev), 300.0),
        "two-way tie": lambda: torch.zeros(ROWS, V, device=dev).scatter_(
            1, torch.randint(0, V, (ROWS, 2), device=dev), 1.0),
    }
    overall = 0.0
    hits_total = 0
    for name, gen in cases.items():
        mx, hits, tries = -1.0, 0, 0
        for _ in range(30):
            logits = gen()
            for temp in (0.7, 1.0, 0.5):
                temps = torch.full((ROWS,), temp, device=dev, dtype=torch.float32)
                q = softmax_temp_flashinfer(
                    logits=logits, temperatures=temps, rows_per_request=1
                )
                tries += 1
                m = float(q.max().item())
                mx = max(mx, m)
                if (q > 1.0).any().item():
                    hits += 1
        overall = max(overall, mx)
        hits_total += hits
        print(f"   {name:14s} tries={tries:3d}  max(q)={mx:.9f}  q>1.0 的批次数={hits}")
    print(f"   ⇒ 全局 max(q) = {overall:.9f}   q>1.0 命中批次合计 = {hits_total}")
    return overall, hits_total


def part_b(dev, q_values):
    print("=" * 72)
    print("B) q>1.0 时 chain_speculative_sampling_triton 的接受长度（p=0.999, coin=0.5）")
    B, SLOTS, V = 1, 6, 8
    gamma = SLOTS - 1
    out = []
    for Q in q_values:
        cand = torch.arange(SLOTS, device=dev, dtype=torch.int32).unsqueeze(0)
        idx = torch.arange(SLOTS, device=dev, dtype=torch.int32).unsqueeze(0)
        tp = torch.zeros((B, gamma, V), device=dev, dtype=torch.float32)
        dp = torch.zeros((B, gamma, V), device=dev, dtype=torch.float32)
        for s in range(gamma):
            tok = int(cand[0, s + 1])
            tp[0, s, tok] = 0.999
            dp[0, s, tok] = Q
        tp[0, :, 0] = 0.001
        dp[0, :, 0] = max(0.0, 1.0 - Q)
        uni = torch.full((B, gamma), 0.5, device=dev, dtype=torch.float32)
        uni_f = torch.full((B,), 0.5, device=dev, dtype=torch.float32)
        predicts = torch.full((B, SLOTS), -1, device=dev, dtype=torch.int32)
        acc_idx = torch.full((B, SLOTS), -1, device=dev, dtype=torch.int32)
        acc_num = torch.zeros((B,), device=dev, dtype=torch.int32)
        chain_speculative_sampling_triton(
            predicts=predicts, accept_index=acc_idx, accept_token_num=acc_num,
            candidates=cand, retrive_index=idx,
            retrive_next_token=idx, retrive_next_sibling=idx,
            uniform_samples=uni, uniform_samples_for_final_sampling=uni_f,
            target_probs=tp, draft_probs=dp,
            threshold_single=1.0, threshold_acc=1.0, deterministic=True,
        )
        torch.cuda.synchronize()
        n = int(acc_num[0].item())
        out.append((Q, n))
        flag = "  ← 缺陷命中（接受被清零）" if (Q > 1.0 and n == 0) else ""
        print(f"   q={Q:<12.7f}  ⇒  accept_token_num = {n} / {gamma}{flag}")
    return out


def main():
    dev = torch.device("cuda", 0)
    print("device:", torch.cuda.get_device_name(0))
    a = part_a(dev)
    b = part_b(dev, [1.0, 0.999999, 1.0000001, 1.000006, 1.00002, 1.01])
    print("=" * 72)
    base = dict(b)[1.0]
    worst = min(n for q, n in b if q > 1.0)
    print("实测结论（仅陈述数字）:")
    print(f"  A: softmax 全局 max(q)={a[0]:.9f}, q>1.0 批次={a[1]}" if a else "  A: n/a")
    print(f"  B: q=1.0 接受 {base}/{5};  q>1.0 最小接受 = {worst}/5  ⇒ 损失 {base-worst}")


if __name__ == "__main__":
    main()
