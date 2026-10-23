#!/usr/bin/env bash
# =============================================================================
# apply / rollback : sglang-dspark-segment-probe
#   Gated (SGLANG_DSV41_PROBE_SEGMENTS=1) decode-step segment timing probe for
#   the DSV4.1 dspark deployment.  Default OFF => zero behaviour change.
#
# 目标树 (默认 site = 运行中的代码):
#   site : /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages
#   repo : /mnt/SSD01/models/.llm-env/Lsglang/lsglang/python
#
# 用法:
#   ./apply_sglang_dspark_segment_probe.sh                 # 应用到 site (幂等)
#   ./apply_sglang_dspark_segment_probe.sh --tree both     # site + repo
#   ./apply_sglang_dspark_segment_probe.sh --dry-run       # 只做 patch --dry-run
#   ./apply_sglang_dspark_segment_probe.sh --status        # 查看是否已应用
#   ./apply_sglang_dspark_segment_probe.sh --rollback      # 还原 (含 site+repo)
#   ./rollback_sglang_dspark_segment_probe.sh              # 等价于 --rollback
#
# 备份: 每个被改文件 -> $DIR/<扁平名>.orig-probe.<tree>  (可多次覆盖, 内容为改前原文)
# 失败自动还原; 应用后 py_compile 校验; 生效标记 = [seg-probe] (可 grep)
# =============================================================================
set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-dspark-segment-probe.patch"
PY="${PY:-/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python}"
TREE_SITE="${TREE_SITE:-/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages}"
TREE_REPO="${TREE_REPO:-/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python}"
MARK="[seg-probe]"

MODIFIED=(
  "sglang/srt/layers/moe/fused_moe_triton/layer.py"
  "sglang/srt/speculative/dspark_components/dspark_draft.py"
  "sglang/kernels/ops/speculative/dspark/markov_sample.py"
  "sglang/kernels/ops/speculative/dspark/dspark_accept.py"
  "sglang/srt/speculative/dspark_components/dspark_worker_v2.py"
)
NEWFILE="sglang/srt/utils/dsv41_segment_probe.py"

flat() { printf '%s' "${1//\//__}"; }
tree_path() { case "$1" in site) printf '%s' "$TREE_SITE";; repo) printf '%s' "$TREE_REPO";; esac; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

MODE="apply"; TREES=("site")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) MODE="dryrun" ;;
    --rollback|-R) MODE="rollback" ;;
    --status) MODE="status" ;;
    --tree) shift; case "${1:-}" in
        site) TREES=(site) ;; repo) TREES=(repo) ;; both) TREES=(site repo) ;;
        *) echo "ERROR: --tree 只接受 site|repo|both" >&2; exit 2 ;; esac ;;
    --both) TREES=(site repo) ;;
    --help|-h) usage; exit 0 ;;
    *) echo "未知参数: $1 (--help)" >&2; exit 2 ;;
  esac
  shift
done

applied_in() {  # $1=tree root
  grep -qF "$MARK" "$1/$NEWFILE" 2>/dev/null && return 0 || return 1
}

do_status() {
  local t root
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"
    if applied_in "$root"; then echo "[$t] APPLIED   ($root)"
    else echo "[$t] not applied ($root)"; fi
    for rel in "${MODIFIED[@]}"; do
      local b="$DIR/$(flat "$rel").orig-probe.$t"
      [[ -f "$b" ]] && echo "        backup: $(basename "$b")"
    done
  done
}

do_rollback() {
  local t root rel b
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"
    for rel in "${MODIFIED[@]}"; do
      b="$DIR/$(flat "$rel").orig-probe.$t"
      if [[ -f "$b" ]]; then
        cp -p "$b" "$root/$rel"
        echo "[$t] restored $rel"
      elif grep -qF "$MARK" "$root/$rel" 2>/dev/null; then
        # patched but no backup: reverse the patch
        if patch -R -p1 -d "$root" --dry-run < "$PATCH" >/dev/null 2>&1; then
          patch -R -p1 -d "$root" < "$PATCH" >/dev/null && echo "[$t] reverted $rel"
        else
          echo "[$t] WARN patched but no backup and reverse patch fails: $rel" >&2
        fi
      else
        echo "[$t] not patched, skip: $rel"
      fi
    done
    if [[ -f "$root/$NEWFILE" && ! -f "$DIR/$(flat "$NEWFILE").orig-probe.$t" ]]; then
      rm -f "$root/$NEWFILE" && echo "[$t] removed $NEWFILE"
    fi
  done
  echo "rollback done (trees: ${TREES[*]})"
}

do_apply() {
  local dry="$1"
  [[ -f "$PATCH" ]] || { echo "ERROR: patch missing: $PATCH" >&2; exit 1; }
  local t root
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"
    [[ -d "$root/sglang" ]] || { echo "ERROR: 不是 sglang 包根: $root" >&2; exit 1; }
    if applied_in "$root"; then echo "[$t] ALREADY APPLIED — skip ($root)"; continue; fi
    if ! patch -p1 -d "$root" --dry-run < "$PATCH" >/dev/null; then
      echo "[$t] ERROR: patch --dry-run 失败, 不应用" >&2
      patch -p1 -d "$root" --dry-run < "$PATCH" || true
      exit 1
    fi
    echo "[$t] dry-run OK"
    if [[ "$dry" == "1" ]]; then continue; fi
    # backup
    local rel
    for rel in "${MODIFIED[@]}"; do
      cp -p "$root/$rel" "$DIR/$(flat "$rel").orig-probe.$t" 2>/dev/null \
        || { echo "[$t] ERROR: backup 失败 $rel" >&2; exit 1; }
    done
    if ! patch -p1 -d "$root" < "$PATCH"; then
      echo "[$t] ERROR: 应用失败, 自动还原" >&2
      for rel in "${MODIFIED[@]}"; do
        cp -p "$DIR/$(flat "$rel").orig-probe.$t" "$root/$rel"
      done
      [[ -f "$root/$NEWFILE" && ! -f "$DIR/$(flat "$NEWFILE").orig-probe.$t" ]] && rm -f "$root/$NEWFILE"
      exit 1
    fi
    # syntax gate
    local -a pyfiles=("$root/$NEWFILE")
    for rel in "${MODIFIED[@]}"; do pyfiles+=("$root/$rel"); done
    if ! "$PY" -m py_compile "${pyfiles[@]}" 2>/dev/null; then
      echo "[$t] ERROR: py_compile 失败, 自动还原" >&2
      for rel in "${MODIFIED[@]}"; do
        cp -p "$DIR/$(flat "$rel").orig-probe.$t" "$root/$rel"
      done
      rm -f "$root/$NEWFILE"
      exit 1
    fi
    echo "[$t] APPLIED + py_compile OK"
  done
  echo "gate: SGLANG_DSV41_PROBE_SEGMENTS=1 (default off)"
}

case "$MODE" in
  status)   do_status ;;
  rollback) do_rollback ;;
  dryrun)   do_apply 1 ;;
  apply)    do_apply 0 ;;
esac
