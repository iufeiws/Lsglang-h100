#!/usr/bin/env bash
# =============================================================================
# apply / rollback : sglang-dsv41-segment-probe-rawrows
#   Gated extension of the DSV4.1 segment probe: SGLANG_DSV41_PROBE_RAW=1 adds
#   the per-step (per-row) values of each window to the window record, so
#   per-step p50/p95 can be computed over any sub-interval of the run WITHOUT
#   shrinking SGLANG_DSV41_PROBE_WINDOW (the sync cadence -- and therefore the
#   measured object -- is unchanged).  Default OFF => the window record is
#   byte-identical to the unextended probe.
#
# 前置条件: 先应用 sglang-dspark-segment-probe.patch (本补丁改的是它新增的模块)。
#
# 目标树 (默认 site = 运行中的代码):
#   site : /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages
#   repo : /mnt/SSD01/models/.llm-env/Lsglang/lsglang/python
#
# 用法:
#   ./apply_sglang_dsv41_segment_probe_rawrows.sh            # 应用到 site (幂等)
#   ./apply_sglang_dsv41_segment_probe_rawrows.sh --tree both
#   ./apply_sglang_dsv41_segment_probe_rawrows.sh --dry-run
#   ./apply_sglang_dsv41_segment_probe_rawrows.sh --status
#   ./apply_sglang_dsv41_segment_probe_rawrows.sh --rollback
#   ./rollback_sglang_dsv41_segment_probe_rawrows.sh
#
# 备份: .orig-segraw.<tree>  |  失败自动还原  |  py_compile 校验
# 生效标记 = [seg-probe-raw] (可 grep)
# =============================================================================
set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-dsv41-segment-probe-rawrows.patch"
PY="${PY:-/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python}"
TREE_SITE="${TREE_SITE:-/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages}"
TREE_REPO="${TREE_REPO:-/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python}"
MARK="[seg-probe-raw]"
SUF="orig-segraw"

MODIFIED=(
  "sglang/srt/utils/dsv41_segment_probe.py"
)

flat() { printf '%s' "${1//\//__}"; }
tree_path() { case "$1" in site) printf '%s' "$TREE_SITE";; repo) printf '%s' "$TREE_REPO";; esac; }

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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

applied_in() { grep -qF "$MARK" "$1/${MODIFIED[0]}" 2>/dev/null; }

do_status() {
  local t root rel b
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"
    rel="${MODIFIED[0]}"
    if [[ ! -f "$root/$rel" ]]; then echo "[$t] module MISSING ($root/$rel) — 先应用 segment-probe"; continue; fi
    if applied_in "$root"; then echo "[$t] APPLIED   ($root) markers=$(grep -cF "$MARK" "$root/$rel")"
    else echo "[$t] not applied ($root) markers=0"; fi
    b="$DIR/$(flat "$rel").$SUF.$t"
    [[ -f "$b" ]] && echo "        backup: $(basename "$b")"
  done
}

do_rollback() {
  local t root rel b
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"; rel="${MODIFIED[0]}"
    b="$DIR/$(flat "$rel").$SUF.$t"
    if [[ -f "$b" ]]; then
      cp -p "$b" "$root/$rel" && echo "[$t] restored $rel (markers=$(grep -cF "$MARK" "$root/$rel" || true))"
    elif grep -qF "$MARK" "$root/$rel" 2>/dev/null; then
      if patch -R -p1 -d "$root" --dry-run < "$PATCH" >/dev/null 2>&1; then
        patch -R -p1 -d "$root" < "$PATCH" >/dev/null && echo "[$t] reverted $rel"
      else
        echo "[$t] WARN patched but reverse patch fails: $rel" >&2
      fi
    else
      echo "[$t] not patched, skip: $rel"
    fi
  done
  echo "rollback done (trees: ${TREES[*]})"
}

do_apply() {
  local dry="$1" t root rel
  [[ -f "$PATCH" ]] || { echo "ERROR: patch missing: $PATCH" >&2; exit 1; }
  for t in "${TREES[@]}"; do
    root="$(tree_path "$t")"; rel="${MODIFIED[0]}"
    [[ -f "$root/$rel" ]] || { echo "[$t] ERROR: 缺少 $rel —— 请先应用 sglang-dspark-segment-probe" >&2; exit 1; }
    if applied_in "$root"; then echo "[$t] ALREADY APPLIED — skip ($root)"; continue; fi
    if ! patch -p1 -d "$root" --dry-run < "$PATCH" >/dev/null; then
      echo "[$t] ERROR: patch --dry-run 失败, 不应用" >&2
      patch -p1 -d "$root" --dry-run < "$PATCH" || true
      exit 1
    fi
    echo "[$t] dry-run OK"
    [[ "$dry" == "1" ]] && continue
    cp -p "$root/$rel" "$DIR/$(flat "$rel").$SUF.$t" || { echo "[$t] ERROR: backup 失败" >&2; exit 1; }
    if ! patch -p1 -d "$root" < "$PATCH"; then
      echo "[$t] ERROR: 应用失败, 自动还原" >&2
      cp -p "$DIR/$(flat "$rel").$SUF.$t" "$root/$rel"; exit 1
    fi
    if ! "$PY" -m py_compile "$root/$rel" 2>/dev/null; then
      echo "[$t] ERROR: py_compile 失败, 自动还原" >&2
      cp -p "$DIR/$(flat "$rel").$SUF.$t" "$root/$rel"; exit 1
    fi
    echo "[$t] APPLIED + py_compile OK ($(grep -cF "$MARK" "$root/$rel") markers)"
  done
  echo "gate: SGLANG_DSV41_PROBE_RAW=1 (default off)"
}

case "$MODE" in
  status)   do_status ;;
  rollback) do_rollback ;;
  dryrun)   do_apply 1 ;;
  apply)    do_apply 0 ;;
esac
