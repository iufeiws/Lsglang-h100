#!/usr/bin/env bash
# 应用补丁: sglang reject_sampling q 容差
#   q ∈ (0, 1+EPS] 视为合法概率并 clamp 到 [0,1]（q>1 等价于 q=1），
#   不再把 q 略微 >1 当作非法概率而把接受长度清零。
#   EPS = Q_PROB_TOL_DEFAULT = 1e-2（依据见 docs/reject-sampling-q-fix.md）
#
# 用法:
#   ./apply_sglang_reject_sampling_q_tolerance.sh            # 应用（幂等）
#   ./apply_sglang_reject_sampling_q_tolerance.sh --dry-run   # 只检查能否干净应用，不改文件
#
# 环境变量: ENV_PREFIX / SITE 可覆盖安装前缀
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-reject-sampling-q-tolerance.patch"
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

[[ -f "$PATCH" ]] || { echo "ERROR: 找不到补丁 $PATCH" >&2; exit 1; }
[[ -f "$TARGET" ]] || { echo "ERROR: 找不到目标 $TARGET" >&2; exit 1; }

if grep -qF -- "$MARK" "$TARGET"; then
  echo "ALREADY APPLIED: $TARGET 已含标记 '$MARK'"
  exit 0
fi

# 已存在的备份：仅当它与当前文件字节一致时才复用（rollback 之后重放的情形）
if [[ -f "$ORIG" ]]; then
  if cmp -s "$ORIG" "$TARGET"; then
    echo "note: 复用已有备份 $ORIG（与当前文件一致）"
  else
    echo "ERROR: $ORIG 存在但与当前文件不同；请人工确认（可能需要先手工删除该备份）" >&2
    exit 1
  fi
fi

# 先在暂存目录里打补丁 + 语法检查，全部通过后才原子替换线上文件：
# 这样线上文件不会出现"改了一半"的中间状态（正在跑的服务即使重读源码也只会读到完整文件）。
STAGE="$(mktemp -d "$SITE/.qfix_stage.XXXXXX")"
STAGE_REL="$STAGE/$REL"
mkdir -p "$(dirname "$STAGE_REL")"
cp -p "$TARGET" "$STAGE_REL"
cleanup() { rm -rf "$STAGE"; }

if [[ $DRY_RUN == 1 ]]; then
  trap cleanup EXIT
  echo "DRY RUN: 不会修改 $TARGET"
  if patch -p1 -d "$STAGE" --dry-run < "$PATCH"; then
    echo "DRY RUN OK: 补丁可干净应用"
    exit 0
  fi
  echo "ERROR: DRY RUN 失败（补丁与当前文件不匹配）" >&2
  exit 1
fi

if ! patch -p1 -d "$STAGE" < "$PATCH"; then
  echo "ERROR: patch 应用失败；$TARGET 未被改动" >&2
  cleanup
  exit 1
fi
if ! "$ENV_PREFIX/bin/python" -m py_compile "$STAGE_REL"; then
  echo "ERROR: 补丁后语法检查失败；$TARGET 未被改动" >&2
  cleanup
  exit 1
fi
if ! grep -qF -- "$MARK" "$STAGE_REL"; then
  echo "ERROR: 暂存文件缺少生效标记；$TARGET 未被改动" >&2
  cleanup
  exit 1
fi

# 备份原始文件（供 rollback 使用）
if [[ ! -f "$ORIG" ]]; then
  cp -p "$TARGET" "$ORIG"
  echo "backup : $ORIG"
fi

# 原子替换
"$ENV_PREFIX/bin/python" - "$STAGE_REL" "$TARGET" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY
cleanup

# 安装后自检：语法 + 标记。失败则自动从 .orig 还原。
if ! "$ENV_PREFIX/bin/python" -m py_compile "$TARGET" || ! grep -qF -- "$MARK" "$TARGET"; then
  echo "ERROR: 安装后自检失败，自动还原" >&2
  "$ENV_PREFIX/bin/python" - "$ORIG" "$TARGET" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY
  rm -f "$ORIG"
  exit 1
fi

echo "APPLIED: $TARGET"
grep -nF -- "$MARK" "$TARGET" | sed 's/^/  /'
echo "回滚: $DIR/rollback_sglang_reject_sampling_q_tolerance.sh"
