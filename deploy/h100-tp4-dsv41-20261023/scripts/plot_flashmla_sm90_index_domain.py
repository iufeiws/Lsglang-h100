#!/usr/bin/env python3
"""Plot the KV byte offset produced by splitkv_mla.cuh:538-539 for a range of
`token_index` values.

Pure arithmetic replication of the kernel lines below (no GPU, no torch):

    block_index      = token_index == -1 ? 0 : (int)((uint32_t)token_index / (uint32_t)page_block_size)
    rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size
    gK_base          = k_ptr + block_index * stride_kv_block
                              + rel_idx_in_block * (HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16))

with the SM90 V4 layout (HEAD_DIM_NOPE=448, HEAD_DIM_ROPE=64 -> 576 B/token,
BYTES_PER_TOKEN=584) and page_block_size = 256 as used by the DSV4.1 deployment.

Output: docs/figs/flashmla_sm90_index_domain.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/SSD01/models/.llm-env/Lsglang/docs/figs/flashmla_sm90_index_domain.png"

PAGE_BLOCK_SIZE = 256          # kv.size(1) passed as page_block_size
BYTES_PER_TOKEN = 584          # KVCacheFormat<V4>::BYTES_PER_TOKEN
DATA_PER_TOKEN = 576           # HEAD_DIM_NOPE(448) + 2*HEAD_DIM_ROPE(64)
STRIDE_KV_BLOCK = PAGE_BLOCK_SIZE * BYTES_PER_TOKEN   # contiguous page stride
NUM_PAGES = 64                 # illustrative pool size: 64 pages = 16384 tokens
POOL_BYTES = NUM_PAGES * STRIDE_KV_BLOCK

UINT32_MAX = 0xFFFFFFFF


def offset_of(token_index: int) -> tuple[int, int, int]:
    """Return (block_index, rel_idx_in_block, byte_offset) as the kernel computes it."""
    if token_index == -1:
        block_index = 0
    else:
        block_index = (token_index & UINT32_MAX) // PAGE_BLOCK_SIZE
    rel = (token_index & UINT32_MAX) % PAGE_BLOCK_SIZE
    return block_index, rel, block_index * STRIDE_KV_BLOCK + rel * DATA_PER_TOKEN


CASES = [
    ("0\n(first valid)", 0),
    ("511\n(last token in page 0)", 511),
    ("16383\n(last token, 64 pages)", NUM_PAGES * PAGE_BLOCK_SIZE - 1),
    ("16384\n= num_pages*page_block_size", NUM_PAGES * PAGE_BLOCK_SIZE),
    ("-1\n(sentinel)", -1),
    ("-2", -2),
    ("-123456", -123456),
    ("2147483647\n(INT_MAX)", 2147483647),
    ("-2147483648\n(INT_MIN)", -2147483648),
]

labels, offsets, pages, rels = [], [], [], []
for label, idx in CASES:
    blk, rel, off = offset_of(idx)
    labels.append(label)
    offsets.append(off)
    pages.append(blk)
    rels.append(rel)

# Morandi palette
C_IN = "#8A9A8B"        # within the pool
C_OUT = "#B08477"       # beyond the pool
C_SENT = "#7B8CA6"      # the -1 sentinel path
colors = [
    C_IN if off < POOL_BYTES else C_OUT for off in offsets
]
colors[4] = C_SENT      # -1 case

fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=170)
ypos = range(len(labels))
ax.barh(list(ypos), offsets, color=colors, height=0.62, zorder=3)

ax.set_yticks(list(ypos))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=1e5)
ax.set_xlabel(
    "KV byte offset from k_ptr  =  page_index * (page_block_size * 584 B)"
    "  +  offset_in_page * 576 B",
    fontsize=9.5,
)
ax.set_ylabel("token_index value", fontsize=9.5)
ax.set_title(
    "SM90 sparse decode (splitkv_mla.cuh:538-539): byte offset produced per token_index\n"
    "page_block_size = 256, V4 layout 576+8 = 584 B/token, illustrative pool = 64 pages"
    " (16384 tokens = 9.57 MB)",
    fontsize=10.5,
)

ax.axvline(POOL_BYTES, color="#6E6A63", linestyle="--", linewidth=1.4, zorder=4)
ax.annotate(
    "end of the illustrative pool\n(64 pages = 9,568,256 B)",
    xy=(POOL_BYTES, 1.45),
    xytext=(POOL_BYTES * 2.6, 1.5),
    fontsize=8.5,
    color="#6E6A63",
    va="center",
    arrowprops=dict(arrowstyle="->", color="#6E6A63", linewidth=1.0),
)

for y, off, blk, rel in zip(ypos, offsets, pages, rels):
    ax.annotate(
        f"page {blk}, row {rel}  ->  {off:,} B",
        xy=(max(off, 1.0), y),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.2,
        color="#4A4743",
    )

ax.set_xlim(1, max(offsets) * 12)
ax.grid(axis="x", color="#D8D4CE", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#B9B4AC")
ax.spines["bottom"].set_color("#B9B4AC")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=C_IN, label="offset < pool bytes"),
    plt.Rectangle((0, 0), 1, 1, color=C_SENT, label="-1 sentinel (remapped to page 0)"),
    plt.Rectangle((0, 0), 1, 1, color=C_OUT, label="offset >= pool bytes"),
]
ax.legend(
    handles=handles,
    fontsize=8.5,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=3,
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
for label, idx, off, blk, rel in zip(labels, [c[1] for c in CASES], offsets, pages, rels):
    print(f"{label.splitlines()[0]:>12}  page={blk:<9} row={rel:<4} offset={off:>16,} B")
