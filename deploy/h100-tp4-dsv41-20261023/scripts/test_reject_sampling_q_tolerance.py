#!/usr/bin/env python
"""单元测试：DSPARK 拒绝采样路径的 q 容差修复（单卡；不加载模型/权重）。

被测对象
  sglang/kernels/ops/speculative/reject_sampling.py : chain_speculative_sampling_triton
    :71   block 验证  q_is_prob = (q > 0.0) & (q <= 1.0)      → 修复点 A（判定+clamp）
    :148  普通验证    q_is_prob = (q > 0.0) & (q <= 1.0)      → 修复点 B（判定+clamp）
    :194  residual pass 1  q_val 越界清零                     → 修复点 C（改 clamp）
    :222  residual pass 2  q_val 越界清零                     → 修复点 D（改 clamp）

修复语义（EPS = Q_PROB_TOL_DEFAULT = 1e-2）：
    q ∈ (0, 1+EPS]  → 视为合法概率，算术前 clamp 到 [0,1]（q>1 等价于 q=1）
    q ≤ 0 / NaN / ±inf / q > 1+EPS → 维持原语义（拒绝；residual 中当作 0）

用法
  # 修复前基线（先用 rollback 脚本还原 site-packages）
  CUDA_VISIBLE_DEVICES=<GPU3-UUID> python test_reject_sampling_q_tolerance.py --mode pre  --dump pre.json
  # 修复后
  CUDA_VISIBLE_DEVICES=<GPU3-UUID> python test_reject_sampling_q_tolerance.py --mode post --dump post.json
  # 修复前/后对照（纯 CPU，不需要 GPU）
  python test_reject_sampling_q_tolerance.py --compare pre.json post.json
  # 预检：直接按路径加载补丁仓里的暂存文件，不碰 site-packages
  CUDA_VISIBLE_DEVICES=<GPU3-UUID> python test_reject_sampling_q_tolerance.py --mode post \
      --module-file /path/to/staged/sglang/kernels/ops/speculative/reject_sampling.py

单卡约束：脚本要求 CUDA_VISIBLE_DEVICES 已把可见设备限制为 1 张（以免误落到别的卡）；
用 --allow-any-gpu 可解除该保护（不推荐）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

MARKER = "# [ENV-PATCH] reject_sampling q-tolerance"
EPS_EXPECTED = 1e-2  # 与 Q_PROB_TOL_DEFAULT 及补丁标记中的 EPS=1e-2 一致
MARKER_EPS_RE = re.compile(re.escape(MARKER) + r"\s*EPS=([0-9.eE+-]+)")

# 本机部署（DeepSeek-V4.1-Flash-1017, DSPARK）的 vocab，用于 EPS 依据： (V-1)*2**-24
DEPLOY_VOCAB = 129280

NAN = float("nan")
INF = float("inf")


# --------------------------------------------------------------------------- #
# 载入被测模块
# --------------------------------------------------------------------------- #
def load_kernel(module_file: Optional[str]):
    if module_file:
        # 用显式 loader：这样 .orig/.pristine 这类非 .py 后缀也能被加载，
        # 便于在不改 site-packages 的情况下直接对比修复前/后两份源码。
        loader = importlib.machinery.SourceFileLoader("reject_sampling_under_test", module_file)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None:
            raise RuntimeError(f"cannot load module from {module_file}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["reject_sampling_under_test"] = mod
        loader.exec_module(mod)
        return mod.chain_speculative_sampling_triton, os.path.abspath(module_file), mod
    from sglang.kernels.ops.speculative.reject_sampling import (  # noqa: PLC0415
        chain_speculative_sampling_triton,
    )
    import sglang.kernels.ops.speculative.reject_sampling as m  # noqa: PLC0415

    return chain_speculative_sampling_triton, os.path.abspath(m.__file__), m


def source_state(src_path: str) -> Tuple[bool, Optional[float], str]:
    """返回 (是否已打补丁, 标记里的 EPS, 源码 sha256)。"""
    with open(src_path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    hit = MARKER_EPS_RE.search(text)
    return (MARKER in text), (float(hit.group(1)) if hit else None), sha


# --------------------------------------------------------------------------- #
# 运行器：本测试只走生产入口 chain_speculative_sampling_triton
# --------------------------------------------------------------------------- #
def run_chain(fn, *, tp, dp, cand, idx, uni, uni_f, block: bool = False):
    """返回 (accept_token_num[B], predicts[B, slots])。"""
    B, slots = cand.shape
    dev = tp.device
    predicts = torch.full((B, slots), -1, dtype=torch.int32, device=dev)
    acc_idx = torch.full((B, slots), -1, dtype=torch.int32, device=dev)
    acc_num = torch.full((B,), -1, dtype=torch.int32, device=dev)
    if block:
        from functools import partial

        fn = partial(fn, block_verification=True)
    fn(
        predicts=predicts,
        accept_index=acc_idx,
        accept_token_num=acc_num,
        candidates=cand,
        retrive_index=idx,
        retrive_next_token=idx,
        retrive_next_sibling=idx,
        uniform_samples=uni,
        uniform_samples_for_final_sampling=uni_f,
        target_probs=tp,
        draft_probs=dp,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    torch.cuda.synchronize()
    return acc_num, predicts


def _empty_case(B, gamma, V, dev):
    # TargetProbs 有 num_slots = gamma + 1 行（:169-172 注释），DraftProbs 只有 gamma 行
    tp = torch.zeros((B, gamma + 1, V), dtype=torch.float32, device=dev)
    dp = torch.zeros((B, gamma, V), dtype=torch.float32, device=dev)
    cand = torch.zeros((B, gamma + 1), dtype=torch.int32, device=dev)
    # RetriveIndex 存的是 Predicts 的"全局扁平下标"（kernel 用
    # Predicts + last_accepted_global_idx 寻址）：必须每行错开 slots，否则 B>1 时
    # 所有 program 会写同一批槽位、互相覆盖（predicts 内容变成写入竞争的结果）。
    slots = gamma + 1
    local = torch.arange(slots, device=dev).unsqueeze(0)
    base = (torch.arange(B, device=dev) * slots).unsqueeze(1)
    idx = (base + local).to(torch.int32).contiguous()
    return tp, dp, cand, idx


def chain_case(dev, *, gamma: int, V: int, q: float, p: float, coin: float,
               coin_final: float, block: bool = False, token: int = 1):
    """每条链每个 slot 都 draft 同一个 token；p/q 每行相同 ⇒ 要么全接受要么第 1 步就拒绝。"""
    tp, dp, cand, idx = _empty_case(1, gamma, V, dev)
    cand[0, 1:] = token
    tp[:, :gamma, token] = p
    tp[:, gamma, token] = p  # 全接受时 final sampling 读 row gamma
    dp[:, :, token] = q
    uni = torch.full((1, gamma), coin, dtype=torch.float32, device=dev)
    uni_f = torch.full((1,), coin_final, dtype=torch.float32, device=dev)
    return dict(tp=tp, dp=dp, cand=cand, idx=idx, uni=uni, uni_f=uni_f, block=block)


def residual_case(dev, *, q_res: float, gamma: int = 5, V: int = 8, coin_final: float = 0.5):
    """构造一次"第 1 步就拒绝"的链，从而强制走 residual 两遍扫描（:194 / :222）。

    - draft 出的 token 是 token 1，其 dp[0,1] = 0.0 ⇒ q ≤ 0 恒拒绝（修复前后一致），
      于是 cur_prob_row = 0，residual 用 target/draft 的第 0 行。
    - residual 行里：token 2 的 p = 0.5、q = q_res；token 3 的 p = 0.4、q = 0.0。
      修复前 q_res = 1.0000001 在 residual 里被清零 ⇒ token 2 贡献 0.5；
      修复后 clamp 到 1.0 ⇒ token 2 贡献 max(0.5-1.0, 0) = 0。
      最终采样 coin_final = 0.5 ⇒ 修复前落到 token 2，修复后落到 token 3。
    """
    tp, dp, cand, idx = _empty_case(1, gamma, V, dev)
    cand[0, 1:] = 1
    tp[0, 0, 2] = 0.5
    tp[0, 0, 3] = 0.4
    dp[0, 0, 2] = q_res
    uni = torch.full((1, gamma), 0.5, dtype=torch.float32, device=dev)
    uni_f = torch.full((1,), coin_final, dtype=torch.float32, device=dev)
    return dict(tp=tp, dp=dp, cand=cand, idx=idx, uni=uni, uni_f=uni_f, block=False)


def battery_case(dev, *, B: int, gamma: int, V: int, seed: int):
    """随机化等价性对照：全部 q ∈ [0,1] 且无 NaN —— 修复前后必须逐位一致。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tp, dp, cand, idx = _empty_case(B, gamma, V, dev)
    tokens = torch.randint(1, V, (B, gamma), generator=g).to(dev)
    p_vals = torch.rand((B, gamma), generator=g).to(dev)
    q_vals = torch.rand((B, gamma), generator=g).to(dev)
    # 1/3 强制易接受、1/3 强制易拒绝，保证覆盖 0 < k < gamma 的中间接受长度
    third = B // 3
    q_vals[:third] = 0.05
    p_vals[third:2 * third] = 0.05
    for s in range(gamma):
        tp[torch.arange(B, device=dev), s, tokens[:, s]] = p_vals[:, s]
        dp[torch.arange(B, device=dev), s, tokens[:, s]] = q_vals[:, s]
    cand[:, 1:] = tokens
    tp[:, gamma, :] = tp[:, 0, :]  # 全接受分支读 row gamma，保持一致
    uni = torch.rand((B, gamma), generator=g).to(dev)
    uni_f = torch.rand((B,), generator=g).to(dev)
    return dict(tp=tp, dp=dp, cand=cand, idx=idx, uni=uni, uni_f=uni_f, block=False)


def law_case(dev, *, B: int, gamma: int, V: int, p: float, q: float, seed: int):
    """接受律检查：每步接受概率应 ≈ min(1, p/q)。用于确认修复没把语义改坏。"""
    tp, dp, cand, idx = _empty_case(B, gamma, V, dev)
    cand[:, 1:] = 1
    tp[:, :, 1] = p
    dp[:, :, 1] = q
    g = torch.Generator(device="cpu").manual_seed(seed)
    uni = torch.rand((B, gamma), generator=g).to(dev)
    uni_f = torch.rand((B,), generator=g).to(dev)
    return dict(tp=tp, dp=dp, cand=cand, idx=idx, uni=uni, uni_f=uni_f, block=False)


def input_sha(case: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in ("tp", "dp", "cand", "idx", "uni", "uni_f"):
        t = case[k].detach().to("cpu").contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    h.update(str(case["block"]).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 用例表
# --------------------------------------------------------------------------- #
# name, q, 期望接受长度（gamma=5）: pre / post
ACCEPT_PATH_CASES: List[Tuple[str, float, int, int]] = [
    # —— 修复点：q 略微 > 1 应从"清零"变为"与 q=1.0 同" ——
    ("q=1.0000001 (1e-7 over)", 1.0000001, 0, 5),
    ("q=1.000006  (6e-6 over)", 1.000006, 0, 5),
    ("q=1.00002   (2e-5 over)", 1.00002, 0, 5),
    ("q=1.01      (1e-2 over)", 1.01, 0, 5),
    # —— 域内参照：修复前后都必须 5/5 ——
    ("q=1.0", 1.0, 5, 5),
    ("q=0.9999999", 0.9999999, 5, 5),
    ("q=0.5", 0.5, 5, 5),
    # —— 超出容差：仍按原语义拒绝（安全性） ——
    ("q=1.0101  (over tol)", 1.0101, 0, 0),
    ("q=1.02    (over tol)", 1.02, 0, 0),
    ("q=1.1     (over tol)", 1.1, 0, 0),
    ("q=2.0     (over tol)", 2.0, 0, 0),
    # —— 非概率：语义必须完全不变 ——
    ("q=0.0", 0.0, 0, 0),
    ("q=-1e-8", -1e-8, 0, 0),
    ("q=-0.5", -0.5, 0, 0),
    ("q=nan", NAN, 0, 0),
    ("q=inf", INF, 0, 0),
]

RESIDUAL_CASES: List[Tuple[str, float, int, int]] = [
    # name, q_res, 期望最终采样 token: pre / post
    ("residual q=1.0 (reference)", 1.0, 3, 3),
    ("residual q=1.0000001", 1.0000001, 2, 3),
]


def fmt_q(q: float) -> str:
    if math.isnan(q):
        return "nan"
    if math.isinf(q):
        return "inf" if q > 0 else "-inf"
    return repr(q)


# --------------------------------------------------------------------------- #
# 测试主体
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pre", "post"], help="断言哪一套期望（pre=修复前基线，post=修复后）")
    ap.add_argument("--module-file", default=None, help="按路径加载被测文件（预检用），默认导入 site-packages 的 sglang")
    ap.add_argument("--dump", default=None, help="把结果写成 JSON，供 --compare 使用")
    ap.add_argument("--compare", nargs=2, metavar=("PRE_JSON", "POST_JSON"), help="对照两份 dump（纯 CPU）")
    ap.add_argument("--gamma", type=int, default=5, help="draft token 数（部署为 5）")
    ap.add_argument("--battery-b", type=int, default=16384, help="随机等价性对照的链数")
    ap.add_argument("--battery-v", type=int, default=32, help="battery 的 vocab")
    ap.add_argument("--law-b", type=int, default=262144, help="接受律检查的链数")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--allow-any-gpu", action="store_true", help="允许可见多张卡（默认要求恰好 1 张）")
    ap.add_argument("--with-softmax-probe", action="store_true",
                    help="额外用 DSPARK 真实 softmax 路径测量 max(q)（需 flashinfer JIT/ninja）")
    args = ap.parse_args(argv)

    if args.compare:
        return compare_dumps(args.compare[0], args.compare[1])

    if not args.mode:
        ap.error("--mode 或 --compare 必须给出其一")

    # —— 单卡保护 ——
    dev_count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dev_count != 1 and not args.allow_any_gpu:
        print(f"ERROR: 可见 {dev_count} 张卡（CUDA_VISIBLE_DEVICES={visible!r}）。"
              f"请用 UUID 限定到单张卡，例如 CUDA_VISIBLE_DEVICES=GPU-ccb33058-8899-4165-97a3-7cc5e661fd25",
              file=sys.stderr)
        return 2
    dev = torch.device("cuda", 0)

    fn, src_path, mod = load_kernel(args.module_file)
    patched, marker_eps, src_sha = source_state(src_path)
    tol_default = getattr(mod, "Q_PROB_TOL_DEFAULT", None)

    print("=" * 78)
    print(f"mode            : {args.mode}")
    print(f"kernel module   : {src_path}")
    print(f"source sha256   : {src_sha}")
    print(f"ENV-PATCH marker: {'present' if patched else 'absent'}"
          f"{'' if marker_eps is None else f' (EPS={marker_eps:g})'}")
    print(f"Q_PROB_TOL_DEFAULT: {tol_default!r}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}  CUDA_VISIBLE_DEVICES={visible!r}")
    print("=" * 78)

    problems: List[str] = []

    # 状态自洽：pre 模式必须跑在未打补丁的文件上，post 模式必须跑在已打补丁的文件上
    if args.mode == "pre" and patched:
        problems.append("mode=pre 但被测文件带有 ENV-PATCH 标记：这是修复后的文件，基线/修复后对照无效")
    if args.mode == "post" and not patched:
        problems.append("mode=post 但被测文件没有 ENV-PATCH 标记：补丁未生效")
    if args.mode == "post":
        if tol_default is None:
            problems.append("修复后应存在 Q_PROB_TOL_DEFAULT 常量")
        elif float(tol_default) != EPS_EXPECTED:
            problems.append(f"Q_PROB_TOL_DEFAULT={tol_default!r} != 期望 {EPS_EXPECTED!r}")
        if marker_eps is None:
            problems.append("补丁标记里没有 EPS=... 数值")
        elif float(marker_eps) != EPS_EXPECTED:
            problems.append(f"标记 EPS={marker_eps!r} 与 Q_PROB_TOL_DEFAULT={tol_default!r} 不一致")

    groups: Dict[str, Any] = {}

    def report(name: str, cases: List[Dict[str, Any]], errors: List[str], metric: str = "accept") -> None:
        print(f"\n--- {name} ---")
        print(f"{'case':34s} {metric:>8s}  {'expected':>8s}  {'verdict':>7s}")
        for c in cases:
            obs = c["observed"]
            exp = c["expected"]
            tag = "ok" if obs == exp else "FAIL"
            print(f"{c['name']:34s} {obs:>8d}  {exp:>8d}  {tag:>7s}")
        for e in errors:
            print(f"    !! {e}")
        groups[name] = {"cases": cases, "errors": errors, "verdict": "PASS" if not errors else "FAIL"}
        print(f"    verdict: {groups[name]['verdict']} ({len(cases)} cases)")

    # ---- 1) 接受路径（:148）----
    cases, errors = [], []
    for name, q, exp_pre, exp_post in ACCEPT_PATH_CASES:
        expect = exp_pre if args.mode == "pre" else exp_post
        c = chain_case(dev, gamma=args.gamma, V=8, q=q, p=0.999, coin=0.5, coin_final=0.5)
        acc, pred = run_chain(fn, **c)
        n = int(acc[0].item())
        cases.append({
            "name": name, "q": fmt_q(q), "accept_num": [n], "expected": expect, "observed": n,
            "expect_pre": exp_pre, "expect_post": exp_post,
            "predicts": [int(x) for x in pred[0].tolist()], "input_sha256": input_sha(c),
        })
        if n != expect:
            errors.append(f"{name}: accept_token_num={n}, 期望 {expect}")
    report("accept_path (non-block, :148)", cases, errors, metric="accept")

    # ---- 2) block 验证路径（:71）----
    cases, errors = [], []
    for name, q, exp_pre, exp_post in ACCEPT_PATH_CASES:
        expect = exp_pre if args.mode == "pre" else exp_post
        c = chain_case(dev, gamma=args.gamma, V=8, q=q, p=0.999, coin=0.5, coin_final=0.5, block=True)
        acc, pred = run_chain(fn, **c)
        n = int(acc[0].item())
        cases.append({
            "name": name, "q": fmt_q(q), "accept_num": [n], "expected": expect, "observed": n,
            "expect_pre": exp_pre, "expect_post": exp_post,
            "predicts": [int(x) for x in pred[0].tolist()], "input_sha256": input_sha(c),
        })
        if n != expect:
            errors.append(f"{name}: accept_token_num={n}, 期望 {expect}")
    report("block_verification (:71)", cases, errors, metric="accept")

    # ---- 3) residual 两遍扫描（:194 / :222）----
    cases, errors = [], []
    for name, q_res, exp_pre, exp_post in RESIDUAL_CASES:
        expect = exp_pre if args.mode == "pre" else exp_post
        c = residual_case(dev, q_res=q_res, gamma=args.gamma)
        acc, pred = run_chain(fn, **c)
        n = int(acc[0].item())
        final_token = int(pred[0, 0].item())  # num_accept=0 ⇒ 预测落在 root 位置
        cases.append({
            "name": name, "q": fmt_q(q_res), "accept_num": [n], "final_token": final_token,
            "expected": expect, "observed": final_token,
            "expect_pre": exp_pre, "expect_post": exp_post,
            "predicts": [int(x) for x in pred[0].tolist()], "input_sha256": input_sha(c),
        })
        if n != 0:
            errors.append(f"{name}: 该构造应恒拒绝（accept=0），实测 {n}")
        if final_token != expect:
            errors.append(f"{name}: final_token={final_token}, 期望 {expect}")
    report("residual scan (:194/:222, final_token)", cases, errors, metric="finaltok")

    # ---- 4) 随机等价性对照（全部 q ∈ [0,1]，无 NaN）----
    bc = battery_case(dev, B=args.battery_b, gamma=args.gamma, V=args.battery_v, seed=args.seed)
    acc, pred = run_chain(fn, **bc)
    acc_list = [int(x) for x in acc.tolist()]
    coverage = {
        "k_eq_0": sum(1 for x in acc_list if x == 0),
        "k_middle": sum(1 for x in acc_list if 0 < x < args.gamma),
        "k_full": sum(1 for x in acc_list if x == args.gamma),
        "mean_accept": sum(acc_list) / len(acc_list),
    }
    print("\n--- battery (q ∈ [0,1] randomized equivalence) ---")
    print(f"B={len(acc_list)} gamma={args.gamma} V={args.battery_v} seed={args.seed}")
    print(f"coverage: {coverage}")
    # kernel 不变量：接受前缀必须等于被接受的 draft token 序列（同时校验本测试的
    # RetriveIndex 全局下标约定没写错）
    inv_bad, tok_bad = [], []
    cand_cpu = bc["cand"].tolist()
    pred_cpu = pred.tolist()
    for b in range(len(acc_list)):
        k = acc_list[b]
        if pred_cpu[b][:k] != cand_cpu[b][1:k + 1]:
            inv_bad.append(b)
        if k < args.gamma and not (0 <= pred_cpu[b][k] < args.battery_v):
            tok_bad.append(b)
    print(f"不变量(接受前缀 == draft 序列): {'OK' if not inv_bad else f'FAIL {len(inv_bad)} 条'}")
    print(f"不变量(final token ∈ [0,V))  : {'OK' if not tok_bad else f'FAIL {len(tok_bad)} 条'}")
    print("这批结果必须与对侧（pre/post）逐位一致 —— 由 --compare 校验")
    groups["battery"] = {
        "B": len(acc_list), "gamma": args.gamma, "V": args.battery_v, "seed": args.seed,
        "accept_num": acc_list,
        "predicts": [[int(x) for x in row] for row in pred.tolist()],
        "input_sha256": input_sha(bc), "coverage": coverage,
        "invariant_prefix_bad": len(inv_bad), "invariant_token_bad": len(tok_bad),
        "verdict": "PASS",
    }
    bat_errors = []
    if coverage["k_eq_0"] == 0 or coverage["k_middle"] == 0 or coverage["k_full"] == 0:
        bat_errors.append("覆盖不足：需要同时出现 k=0 / 0<k<gamma / k=gamma 三种接受长度")
    if inv_bad:
        bat_errors.append(f"不变量失败：{len(inv_bad)} 条链的接受前缀 != draft token 序列（前 10: {inv_bad[:10]}）")
    if tok_bad:
        bat_errors.append(f"不变量失败：{len(tok_bad)} 条链的 final token 越界（前 10: {tok_bad[:10]}）")
    if bat_errors:
        groups["battery"]["verdict"] = "FAIL"
        groups["battery"]["errors"] = bat_errors
        for e in bat_errors:
            print(f"    !! {e}")

    # ---- 5) 接受律检查（p=0.3, q=0.6 ⇒ 每步接受概率 0.5）----
    lc = law_case(dev, B=args.law_b, gamma=args.gamma, V=32, p=0.3, q=0.6, seed=args.seed + 1)
    acc, _ = run_chain(fn, **lc)
    acc_list_law = [int(x) for x in acc.tolist()]
    mean = sum(acc_list_law) / len(acc_list_law)
    p_step = 0.3 / 0.6
    g_ = args.gamma
    # 每步独立以 s 接受、在第 gamma 步截断： P(k) = (1-s)s^k (k<g), P(g) = s^g
    exp_mean = sum(k * ((1 - p_step) * p_step ** k) for k in range(g_)) + g_ * p_step ** g_
    exp_sq = sum(k * k * ((1 - p_step) * p_step ** k) for k in range(g_)) + g_ * g_ * p_step ** g_
    var = max(1e-12, exp_sq - exp_mean ** 2)
    first_rate = sum(1 for x in acc_list_law if x >= 1) / len(acc_list_law)
    sigma_mean = math.sqrt(var) / math.sqrt(len(acc_list_law))
    tol_mean = 4 * sigma_mean
    sigma_first = math.sqrt(0.25 / len(acc_list_law))
    print(f"\n--- acceptance law (p=0.3, q=0.6, step accept prob = min(1,p/q) = {p_step}) ---")
    print(f"B={len(acc_list_law)}  E[accept] = {mean:.4f} (theory {exp_mean:.4f}, ±4σ={tol_mean:.4f})")
    print(f"P(first accepted) = {first_rate:.4f} (theory {p_step:.4f}, ±4σ={4 * sigma_first:.4f})")
    law_errors = []
    if abs(mean - exp_mean) > tol_mean:
        law_errors.append(f"E[accept]={mean:.4f} 偏离理论值 {exp_mean:.4f} 超过 4σ={tol_mean:.4f}")
    if abs(first_rate - p_step) > 4 * sigma_first:
        law_errors.append(f"首步接受率 {first_rate:.4f} 偏离 {p_step} 超过 4σ")
    for e in law_errors:
        print(f"    !! {e}")
    groups["acceptance_law"] = {
        "B": len(acc_list_law), "mean_accept": mean, "expected_mean_accept": exp_mean,
        "tol_4sigma": tol_mean, "first_accept_rate": first_rate, "theory_first": p_step,
        "errors": law_errors, "verdict": "PASS" if not law_errors else "FAIL",
    }
    print(f"    verdict: {groups['acceptance_law']['verdict']}")

    # ---- 6) 可选：真实 softmax 路径的 max(q) ----
    if args.with_softmax_probe:
        print("\n--- softmax probe (DSPARK softmax_temp_flashinfer → flashinfer.sampling.softmax) ---")
        try:
            from sglang.kernels.ops.speculative.dspark.dspark_accept import (  # noqa: PLC0415
                softmax_temp_flashinfer,
            )
            torch.manual_seed(0)
            ROWS, V = 6, 8192
            gens = {
                "N(0,1)": lambda: torch.randn(ROWS, V, device=dev),
                "N(0,5)": lambda: torch.randn(ROWS, V, device=dev) * 5,
                "N(0,20)": lambda: torch.randn(ROWS, V, device=dev) * 20,
                "N(0,100)": lambda: torch.randn(ROWS, V, device=dev) * 100,
                "peaked(+30)": lambda: torch.zeros(ROWS, V, device=dev).scatter_(
                    1, torch.randint(0, V, (ROWS, 1), device=dev), 30.0),
                "peaked(+300)": lambda: torch.full((ROWS, V), -300.0, device=dev).scatter_(
                    1, torch.randint(0, V, (ROWS, 1), device=dev), 300.0),
            }
            probe = {"overall_max": 0.0, "hits": 0, "tries": 0, "per_case": {}}
            for name, gen in gens.items():
                mx, hits, tries = -1.0, 0, 0
                for _ in range(20):
                    logits = gen()
                    for temp in (0.7, 1.0, 0.5):
                        temps = torch.full((ROWS,), temp, device=dev, dtype=torch.float32)
                        q = softmax_temp_flashinfer(logits=logits, temperatures=temps, rows_per_request=1)
                        tries += 1
                        mx = max(mx, float(q.max().item()))
                        if (q > 1.0).any().item():
                            hits += 1
                probe["overall_max"] = max(probe["overall_max"], mx)
                probe["hits"] += hits
                probe["tries"] += tries
                probe["per_case"][name] = {"max_q": mx, "hits": hits, "tries": tries}
                print(f"    {name:14s} tries={tries:3d} max(q)={mx:.9f} q>1.0 批次数={hits}")
            print(f"    ⇒ overall max(q)={probe['overall_max']:.9f}, q>1.0 批次={probe['hits']}/{probe['tries']}")
            groups["softmax_probe"] = dict(probe, verdict="PASS")
        except Exception as exc:  # noqa: BLE001
            print(f"    !! softmax probe 不可用（不影响主体测试）: {exc!r}")
            groups["softmax_probe"] = {"error": repr(exc), "verdict": "SKIP"}

    # ---- 汇总 ----
    for name, g in groups.items():
        for e in g.get("errors", []):
            problems.append(f"[{name}] {e}")

    payload = {
        "mode": args.mode, "module": src_path, "source_sha256": src_sha,
        "marker_present": patched, "marker_eps": marker_eps,
        "q_prob_tol_default": (float(tol_default) if tol_default is not None else None),
        "eps_expected": EPS_EXPECTED,
        "deploy_vocab": DEPLOY_VOCAB,
        "eps_derivation_bound": (DEPLOY_VOCAB - 1) * 2 ** -24,
        "device": torch.cuda.get_device_name(0), "cuda_visible_devices": visible,
        "gamma": args.gamma, "groups": groups, "problems": problems,
    }
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n[written] {os.path.abspath(args.dump)}")

    print("\n" + "=" * 78)
    if problems:
        print(f"RESULT: FAIL ({len(problems)} problem(s))")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("RESULT: PASS")
    return 0


# --------------------------------------------------------------------------- #
# 对照（纯 CPU）
# --------------------------------------------------------------------------- #
def compare_dumps(pre_path: str, post_path: str) -> int:
    with open(pre_path, encoding="utf-8") as fh:
        pre = json.load(fh)
    with open(post_path, encoding="utf-8") as fh:
        post = json.load(fh)

    problems: List[str] = []
    print("=" * 78)
    print(f"pre : {pre['mode']}  marker={pre['marker_present']}  sha={pre['source_sha256'][:16]}")
    print(f"post: {post['mode']}  marker={post['marker_present']}  sha={post['source_sha256'][:16]}")
    print("=" * 78)
    if pre["marker_present"] or not post["marker_present"]:
        problems.append("dump 状态不符：pre 必须是未打补丁文件，post 必须是已打补丁文件")
    if pre["source_sha256"] == post["source_sha256"]:
        problems.append("pre/post 源码 sha256 相同：对照无效")

    # 1) 修复效果：q ∈ (1, 1+EPS] 从 0 → gamma
    print("\n--- 修复效果（accept_token_num: pre → post）---")
    for group in ("accept_path (non-block, :148)", "block_verification (:71)"):
        pre_cases = {c["name"]: c for c in pre["groups"][group]["cases"]}
        post_cases = {c["name"]: c for c in post["groups"][group]["cases"]}
        for name, pc in pre_cases.items():
            qc = post_cases[name]
            a, b = pc["accept_num"][0], qc["accept_num"][0]
            changed = a != b
            expect_change = pc["expect_pre"] != pc["expect_post"]
            status = "ok"
            if changed != expect_change:
                status = "FAIL"
                problems.append(f"[{group}] {name}: pre={a} post={b} 但期望{'有' if expect_change else '无'}变化")
            print(f"{group[:22]:24s} {name:30s} {a} → {b}   {'changed' if changed else 'same':8s} {status}")

    # 2) 等价性：q ≤ 1 与非概率输入必须逐位一致
    print("\n--- 等价性（q ≤ 1 / NaN / ±inf / q ≤ 0：必须逐位一致）---")
    for group in ("accept_path (non-block, :148)", "block_verification (:71)"):
        pre_cases = {c["name"]: c for c in pre["groups"][group]["cases"]}
        post_cases = {c["name"]: c for c in post["groups"][group]["cases"]}
        for name, pc in pre_cases.items():
            if pc["expect_pre"] != pc["expect_post"]:
                continue
            qc = post_cases[name]
            if pc["accept_num"] != qc["accept_num"] or pc["predicts"] != qc["predicts"]:
                problems.append(f"[{group}] {name}: 期望一致但结果不同 "
                                f"pre={pc['accept_num']}/{pc['predicts']} post={qc['accept_num']}/{qc['predicts']}")
                print(f"{name:34s} FAIL")
            else:
                print(f"{name:34s} identical (accept={pc['accept_num']}, predicts={pc['predicts']})")

    # 3) residual：域内 q=1.0 必须一致；越界 q 必须按记录变化
    print("\n--- residual (:194/:222) ---")
    pre_cases = {c["name"]: c for c in pre["groups"]["residual scan (:194/:222, final_token)"]["cases"]}
    post_cases = {c["name"]: c for c in post["groups"]["residual scan (:194/:222, final_token)"]["cases"]}
    for name, pc in pre_cases.items():
        qc = post_cases[name]
        a, b = pc["final_token"], qc["final_token"]
        expect_change = pc["expect_pre"] != pc["expect_post"]
        status = "ok" if (a != b) == expect_change else "FAIL"
        if status == "FAIL":
            problems.append(f"[residual] {name}: pre={a} post={b} 变化不符合预期")
        print(f"{name:34s} final_token {a} → {b}   {'changed' if a != b else 'same':8s} {status}")
    same_ref = pre_cases["residual q=1.0 (reference)"]["final_token"] == post_cases["residual q=1.0 (reference)"]["final_token"]
    if not same_ref:
        problems.append("[residual] q=1.0 参照在 pre/post 之间变化：等价性被破坏")
    if post_cases["residual q=1.0000001"]["final_token"] != post_cases["residual q=1.0 (reference)"]["final_token"]:
        problems.append("[residual] 修复后 q=1.0000001 的行为未与 q=1.0 对齐")
    else:
        print("修复后 residual: q=1.0000001 与 q=1.0 的最终采样结果一致（clamp 生效）")

    # 4) 随机等价性对照（全部 q ∈ [0,1]）
    print("\n--- battery（随机 q ∈ [0,1]，B 条链）---")
    bp, bq = pre["groups"]["battery"], post["groups"]["battery"]
    print(f"B={bp['B']} gamma={bp['gamma']} V={bp['V']} seed={bp['seed']}")
    print(f"coverage pre={bp['coverage']} post={bq['coverage']}")
    if bp["input_sha256"] != bq["input_sha256"]:
        problems.append("[battery] 两次运行的输入张量不一致（sha256 不同），对照无效")
    else:
        print(f"input sha256 一致: {bp['input_sha256'][:16]}")
    if bp["accept_num"] != bq["accept_num"]:
        diff = [i for i, (x, y) in enumerate(zip(bp["accept_num"], bq["accept_num"])) if x != y]
        problems.append(f"[battery] accept_num 有 {len(diff)} 处不同（前 10 个下标 {diff[:10]}）")
        print(f"accept_num: {len(diff)} 处不同  FAIL")
    else:
        print(f"accept_num: {bp['B']}/{bp['B']} 条链逐值相同（含 k=0 / 中间 / 满接受）")
    if bp["predicts"] != bq["predicts"]:
        problems.append("[battery] predicts 有差异")
        print("predicts: 存在差异  FAIL")
    else:
        print(f"predicts  : {bp['B']}×{bp['gamma'] + 1} 个 token 逐值相同")

    # 5) 接受律
    lp, lq = pre["groups"]["acceptance_law"], post["groups"]["acceptance_law"]
    print("\n--- 接受律 ---")
    print(f"E[accept] pre={lp['mean_accept']:.4f} post={lq['mean_accept']:.4f} (theory {lp['expected_mean_accept']:.4f})")
    print(f"首步接受率 pre={lp['first_accept_rate']:.4f} post={lq['first_accept_rate']:.4f} (theory {lp['theory_first']:.4f})")
    for tag, g in (("pre", lp), ("post", lq)):
        if g.get("errors"):
            problems.append(f"[acceptance_law:{tag}] {g['errors']}")

    print("\n" + "=" * 78)
    if problems:
        print(f"COMPARE RESULT: FAIL ({len(problems)} problem(s))")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("COMPARE RESULT: PASS —— 修复消除了 q>1 的接受清零；q ≤ 1 与非概率输入逐位不变；"
          "residual 越界值改为 clamp 并与 q=1.0 对齐")
    return 0


if __name__ == "__main__":
    sys.exit(main())
