#!/usr/bin/env bash
# 把仓库 (61d60cfa) 的 4 个 .py 拷进部署树 site-packages/sglang, 清 __pycache__, 逐份 md5 核对。
# 服务不读仓库工作树: 改动生效的唯一方式是拷进 site-packages。
set -Eeuo pipefail
SRC=/mnt/SSD01/models/.llm-env/Lsglang-h100/python/sglang
DST=/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang
TAG_OLD=d467311a70            # 被替换的版本
FILES=(
  kernels/ops/moe/lk_verify_uva.py
  srt/layers/moe/fused_moe_triton/layer.py
  srt/layers/moe/gpu_expert_cache.py
  srt/utils/lk_moe_env.py
)
for f in "${FILES[@]}"; do
  s="$SRC/$f"; d="$DST/$f"
  [[ -f "$s" ]] || { echo "MISSING SRC $s" >&2; exit 1; }
  [[ -f "$d" ]] || { echo "MISSING DST $d" >&2; exit 1; }
  [[ -f "$d.orig-$TAG_OLD.site" ]] || cp -p "$d" "$d.orig-$TAG_OLD.site"
  cp -f "$s" "$d"
  rm -f "$(dirname "$d")/__pycache__/$(basename "${f%.py}").cpython-312.pyc"
  a=$(md5sum "$s" | cut -d' ' -f1); b=$(md5sum "$d" | cut -d' ' -f1)
  if [[ "$a" == "$b" ]]; then echo "OK   $f  md5=$a"; else echo "FAIL $f src=$a dst=$b" >&2; exit 1; fi
done
echo "--- __pycache__ 残留检查 (应为空) ---"
for f in "${FILES[@]}"; do
  d="$DST/$f"
  ls "$(dirname "$d")/__pycache__/$(basename "${f%.py}").cpython-312.pyc" 2>/dev/null || true
done
echo "deploy done: tag=$TAG_OLD -> 61d60cfa0d"
