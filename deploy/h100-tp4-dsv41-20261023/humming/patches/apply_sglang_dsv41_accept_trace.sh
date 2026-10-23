#!/usr/bin/env bash
# =============================================================================
# apply / rollback : sglang-dsv41-accept-trace
#   Gated (SGLANG_DSV41_PROBE_ACCEPT_TRACE=1) per-step / per-request / per-slot
#   accept trace for the DSV4.1 dspark verify step.
#   Default OFF => zero behaviour change (one python call + one bool test/step).
#
# 目标树 (默认 site = 运行中的代码):
#   site : /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages
#   repo : /mnt/SSD01/models/.llm-env/Lsglang/lsglang/python
#
# 用法:
#   ./apply_sglang_dsv41_accept_trace.sh                 # 应用到 site (幂等)
#   ./apply_sglang_dsv41_accept_trace.sh --tree both     # site + repo
#   ./apply_sglang_dsv41_accept_trace.sh --dry-run       # 只做 patch --dry-run
#   ./apply_sglang_dsv41_accept_trace.sh --status        # 查看是否已应用
#   ./apply_sglang_dsv41_accept_trace.sh --rollback      # 还原 (含 site+repo)
#   ./rollback_sglang_dsv41_accept_trace.sh              # 等价于 --rollback
#
# 备份: 每个被改文件 -> $DIR/<扁平名>.orig-acctrace.<tree>  (内容为改前原文)
# 失败自动还原; 应用后 py_compile 校验; 生效标记 = [accept-trace] (可 grep)
# =============================================================================
set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-dsv41-accept-trace.patch"
PY="${PY:-/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python}"
TREE_SITE="${TREE_SITE:-/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages}"
TREE_REPO="${TREE_REPO:-/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python}"
MARK="[accept-trace]"
SUF="orig-acctrace"

MODIFIED=(
  "sglang/srt/speculative/dspark_components/dspark_observability.py"
)
NEWFILE="sglang/srt/utils/dsv41_accept_trace.py"

flat() { printf '%s' "${1//\//__}"; }
tree_path() { case "$1" in site) printf '%s' "$TREE_SITE";; repo) printf '%s' "$TREE_REPO";; esac; }

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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

applied_in() { grep -qF "$MARK" "$1/$NEWFILE" 2>/dev/null; }

do_status() {
  local t root rel b n
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"
    n="$(grep -cF "$MARK" "$root/$NEWFILE" 2>/dev/null || true)"
    if applied_in "$root"; then echo "[$t] APPLIED   ($root)  markers_in_newfile=$n"
    else echo "[$t] not applied ($root)"; fi
    for rel in "${MODIFIED[@]}"; do
      b="$DIR/$(flat "$rel").$SUF.$t"
      [[ -f "$b" ]] && echo "        backup: $(basename "$b")  markers_in_target=$(grep -cF "$MARK" "$root/$rel" 2>/dev/null || true)"
    done
  done
}

do_rollback() {
  local t root rel b
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"
    for rel in "${MODIFIED[@]}"; do
      b="$DIR/$(flat "$rel").$SUF.$t"
      if [[ -f "$b" ]]; then
        cp -p "$b" "$root/$rel"
        echo "[$t] restored $rel (markers=$(grep -cF "$MARK" "$root/$rel" 2>/dev/null || true))"
      elif grep -qF "$MARK" "$root/$rel" 2>/dev/null; then
        if patch -R -p1 -d "$root" --dry-run < "$PATCH" >/dev/null 2>&1; then
          patch -R -p1 -d "$root" < "$PATCH" >/dev/null && echo "[$t] reverted $rel"
        else
          echo "[$t] WARN patched but no backup and reverse patch fails: $rel" >&2
        fi
      else
        echo "[$t] not patched, skip: $rel"
      fi
    done
    if [[ -f "$root/$NEWFILE" && ! -f "$DIR/$(flat "$NEWFILE").$SUF.$t" ]]; then
      rm -f "$root/$NEWFILE" && echo "[$t] removed $NEWFILE"
    fi
  done
  echo "rollback done (trees: ${TREES[*]})"
}

do_apply() {
  local dry="$1"
  [[ -f "$PATCH" ]] || { echo "ERROR: patch missing: $PATCH" >&2; exit 1; }
  local t root rel
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
    for rel in "${MODIFIED[@]}"; do
      cp -p "$root/$rel" "$DIR/$(flat "$rel").$SUF.$t" 2>/dev/null \
        || { echo "[$t] ERROR: backup 失败 $rel" >&2; exit 1; }
    done
    if ! patch -p1 -d "$root" < "$PATCH"; then
      echo "[$t] ERROR: 应用失败, 自动还原" >&2
      for rel in "${MODIFIED[@]}"; do
        cp -p "$DIR/$(flat "$rel").$SUF.$t" "$root/$rel"
      done
      [[ -f "$root/$NEWFILE" && ! -f "$DIR/$(flat "$NEWFILE").$SUF.$t" ]] && rm -f "$root/$NEWFILE"
      exit 1
    fi
    local -a pyfiles=("$root/$NEWFILE")
    for rel in "${MODIFIED[@]}"; do pyfiles+=("$root/$rel"); done
    if ! "$PY" -m py_compile "${pyfiles[@]}" 2>/dev/null; then
      echo "[$t] ERROR: py_compile 失败, 自动还原" >&2
      for rel in "${MODIFIED[@]}"; do
        cp -p "$DIR/$(flat "$rel").$SUF.$t" "$root/$rel"
      done
      rm -f "$root/$NEWFILE"
      exit 1
    fi
    echo "[$t] APPLIED + py_compile OK ($(grep -cF "$MARK" "$root/$rel") marker(s) in target)"
  done
  echo "gate: SGLANG_DSV41_PROBE_ACCEPT_TRACE=1 (default off)"
}

case "$MODE" in
  status)   do_status ;;
  rollback) do_rollback ;;
  dryrun)   do_apply 1 ;;
  apply)    do_apply 0 ;;
esac
