#!/usr/bin/env bash
# 还原补丁: sglang reject_sampling q 容差（从 .orig 恢复）
#
# 用法:
#   ./rollback_sglang_reject_sampling_q_tolerance.sh            # 还原（幂等）
#   ./rollback_sglang_reject_sampling_q_tolerance.sh --dry-run   # 只报告会做什么，不改文件
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
REL="sglang/kernels/ops/speculative/reject_sampling.py"
TARGET="$SITE/$REL"
ORIG="$TARGET.orig"
MARK="[ENV-PATCH] reject_sampling q-tolerance"

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  "") ;;
  *) echo "用法: $0 [--dry-run]" >&2; exit 2 ;;
esac

[[ -f "$TARGET" ]] || { echo "ERROR: 找不到目标 $TARGET" >&2; exit 1; }

patched=0
grep -qF -- "$MARK" "$TARGET" && patched=1

if [[ $patched == 0 ]]; then
  echo "NOT PATCHED: $TARGET 无标记，无需还原"
  exit 0
fi
if [[ ! -f "$ORIG" ]]; then
  echo "ERROR: $TARGET 带标记但备份 $ORIG 不存在，无法还原" >&2
  exit 1
fi

if [[ $DRY_RUN == 1 ]]; then
  echo "DRY RUN: 会用 $ORIG 覆盖 $TARGET（当前文件不含标记即视为还原成功）"
  diff -u --label "current(patched)" --label "orig(pristine)" "$TARGET" "$ORIG" | head -20 || true
  exit 0
fi

STAGE="$(mktemp -d "$SITE/.qfix_stage.XXXXXX")"
STAGE_REL="$STAGE/$REL"
mkdir -p "$(dirname "$STAGE_REL")"
cp -p "$ORIG" "$STAGE_REL"
cleanup() { rm -rf "$STAGE"; }

if ! "$ENV_PREFIX/bin/python" -m py_compile "$STAGE_REL"; then
  echo "ERROR: 备份文件语法检查失败；$TARGET 未被改动" >&2
  cleanup
  exit 1
fi
if grep -qF -- "$MARK" "$STAGE_REL"; then
  echo "ERROR: 备份文件里也有补丁标记（备份被污染）；$TARGET 未被改动" >&2
  cleanup
  exit 1
fi

"$ENV_PREFIX/bin/python" - "$STAGE_REL" "$TARGET" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY
cleanup

if grep -qF -- "$MARK" "$TARGET"; then
  echo "ERROR: 还原后仍带标记，自检失败" >&2
  exit 1
fi
if ! cmp -s "$ORIG" "$TARGET"; then
  echo "ERROR: 还原后与备份字节不一致" >&2
  exit 1
fi
echo "ROLLED BACK: $TARGET == $ORIG (sha256 $(sha256sum "$TARGET" | cut -c1-16))"
