#!/usr/bin/env python3
"""reject_sampling q 容差修复结果图 (英文标注, 莫兰迪配色). 只画数字, 不作推断.

数据来源: 测试脚本产出的 pre/post JSON dump
  /mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/pre_fixed_harness.json
  /mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/post_fixed_harness.json
输出:
  /mnt/SSD01/models/.llm-env/Lsglang/docs/figs/reject-sampling-q-tolerance-20261023.png
"""
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_SAGE = "#A8B5A0"
M_ROSE = "#C9ADA7"
M_BLUE = "#8E9AAF"
M_SAND = "#D8C3A5"
M_CLAY = "#B5838D"
M_SLATE = "#6D6875"
M_GREY = "#B0AFA8"

EV = "/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance"
OUT = "/mnt/SSD01/models/.llm-env/Lsglang/docs/figs/reject-sampling-q-tolerance-20261023.png"

pre = json.load(open(f"{EV}/pre_fixed_harness.json"))
post = json.load(open(f"{EV}/post_fixed_harness.json"))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6))

# ---- panel A: accept length vs q (single-chain, gamma = 5) ----
gp = {c["name"]: c for c in pre["groups"]["accept_path (non-block, :148)"]["cases"]}
gq = {c["name"]: c for c in post["groups"]["accept_path (non-block, :148)"]["cases"]}
order = ["q=0.5", "q=0.9999999", "q=1.0", "q=1.0000001 (1e-7 over)",
         "q=1.000006  (6e-6 over)", "q=1.00002   (2e-5 over)",
         "q=1.01      (1e-2 over)", "q=1.0101  (over tol)", "q=1.02    (over tol)",
         "q=1.1     (over tol)", "q=2.0     (over tol)"]
labels = ["0.5", "0.9999999", "1.0", "1.0000001", "1.000006", "1.00002",
          "1.01", "1.0101", "1.02", "1.1", "2.0"]
xs = range(len(order))
pre_v = [gp[k]["accept_num"][0] for k in order]
post_v = [gq[k]["accept_num"][0] for k in order]
w = 0.38
ax1.bar([x - w / 2 for x in xs], pre_v, w, color=M_ROSE, edgecolor="#8a6d68", linewidth=0.6, label="pre-fix")
ax1.bar([x + w / 2 for x in xs], post_v, w, color=M_SAGE, edgecolor="#6f7d68", linewidth=0.6, label="post-fix")
for x, v in zip(xs, pre_v):
    ax1.text(x - w / 2, v + 0.07, str(v), ha="center", fontsize=7.5, color="#7a4a55")
    if v == 0:
        ax1.plot([x - w / 2], [0.02], marker="_", ms=9, color="#8a6d68", clip_on=False)
for x, v in zip(xs, post_v):
    ax1.text(x + w / 2, v + 0.07, str(v), ha="center", fontsize=7.5, color="#4f5d49")
    if v == 0:
        ax1.plot([x + w / 2], [0.02], marker="_", ms=9, color="#6f7d68", clip_on=False)
ax1.axvspan(2.5, 6.5, color=M_SAND, alpha=0.35, zorder=0)
ax1.text(4.5, 5.35, r"$q \in (1,\ 1+10^{-2}]$", ha="center", fontsize=9, color="#7a6a4f")
ax1.text(4.5, 4.85, "tolerance band", ha="center", fontsize=8.5, color="#7a6a4f")
ax1.set_xticks(list(xs))
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax1.set_xlabel("draft probability q", fontsize=9.5)
ax1.set_ylabel("accepted draft tokens (of 5)", fontsize=9.5)
ax1.set_title("A. accept length vs q  (single chain, gamma=5, p=0.999, coin=0.5)", fontsize=10)
ax1.set_ylim(0, 6)
ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
ax1.grid(axis="y", color=M_GREY, alpha=0.35, linewidth=0.5)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ---- panel B: residual scan final token ----
rp = {c["name"]: c for c in pre["groups"]["residual scan (:194/:222, final_token)"]["cases"]}
rq = {c["name"]: c for c in post["groups"]["residual scan (:194/:222, final_token)"]["cases"]}
names = ["residual q=1.0 (reference)", "residual q=1.0000001"]
xlab = ["q = 1.0", "q = 1.0000001"]
xs2 = range(len(names))
ax2.bar([x - w / 2 for x in xs2], [rp[k]["final_token"] for k in names], w,
        color=M_ROSE, edgecolor="#8a6d68", linewidth=0.6, label="pre-fix")
ax2.bar([x + w / 2 for x in xs2], [rq[k]["final_token"] for k in names], w,
        color=M_SAGE, edgecolor="#6f7d68", linewidth=0.6, label="post-fix")
for x, k in zip(xs2, names):
    ax2.text(x - w / 2, rp[k]["final_token"] + 0.08, str(rp[k]["final_token"]),
             ha="center", fontsize=8.5, color="#7a4a55")
    ax2.text(x + w / 2, rq[k]["final_token"] + 0.08, str(rq[k]["final_token"]),
             ha="center", fontsize=8.5, color="#4f5d49")
ax2.set_xticks(list(xs2))
ax2.set_xticklabels(xlab, fontsize=8.5)
ax2.set_ylim(0, 4)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_ylabel("final sampled token id (vocab=8)", fontsize=9.5)
ax2.set_title("B. residual scan :194/:222  (forced rejection, coin_final=0.5)", fontsize=10)
ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
ax2.grid(axis="y", color=M_GREY, alpha=0.35, linewidth=0.5)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# ---- panel C: battery accept-length histogram (q ~ U[0,1]) ----
bpre = pre["groups"]["battery"]
bpost = post["groups"]["battery"]
g = pre["groups"]["battery"]["gamma"]
hist_pre = [sum(1 for v in bpre["accept_num"] if v == k) for k in range(g + 1)]
hist_post = [sum(1 for v in bpost["accept_num"] if v == k) for k in range(g + 1)]
xs3 = range(g + 1)
ax3.bar([x - w / 2 for x in xs3], hist_pre, w, color=M_ROSE, edgecolor="#8a6d68",
        linewidth=0.6, label="pre-fix")
ax3.bar([x + w / 2 for x in xs3], hist_post, w, color=M_SAGE, edgecolor="#6f7d68",
        linewidth=0.6, label="post-fix")
ident = bpre["predicts"] == bpost["predicts"] and bpre["accept_num"] == bpost["accept_num"]
ax3.text(0.5, 0.93,
         f"chains compared: {bpre['B']}\n"
         f"predict buffers identical: {ident}\n"
         f"identical tokens: {bpre['B']} x {g + 1} = {bpre['B'] * (g + 1)}",
         transform=ax3.transAxes, ha="center", va="top", fontsize=8.5, color="#4a5568")
ax3.set_xticks(list(xs3))
ax3.set_xlabel("accepted draft tokens (q ~ U[0,1], no NaN)", fontsize=9.5)
ax3.set_ylabel("chains (of 16384)", fontsize=9.5)
ax3.set_title("C. randomized equivalence battery: accept-length histogram", fontsize=10)
ax3.legend(fontsize=8.5, frameon=False, loc="upper right")
ax3.grid(axis="y", color=M_GREY, alpha=0.35, linewidth=0.5)
ax3.set_axisbelow(True)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.suptitle("sglang reject_sampling q-tolerance fix (EPS = 1e-2) — measured on GPU3 (H100), 2026-10-23",
             fontsize=11, color="#3f3a45")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160)
print("written:", OUT)
