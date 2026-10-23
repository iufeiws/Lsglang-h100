#!/usr/bin/env bash
# =============================================================================
# apply_sglang_pool6row_probe.sh   -- GATED (default OFF) 6-row pool contract probe
#
# Gate: LK_POOL_6ROW_PROBE  (1/true/on/yes). Unset or 0 => zero behaviour change:
#   * the CALL block is guarded by `_LK_POOL_6ROW_ON`, a module global that is
#     False unless the env var is set when layer.py is imported => one `if
#     False` test per _cpu_decode call (no CUDA call, no tensor, no file, no
#     print, no sync).
#   * dsv41_pool6row_probe.probe() returns None while the gate is off.
#
# Installs:
#   <tree>/sglang/srt/layers/moe/fused_moe_triton/layer.py   (2 marker blocks
#       spliced in: an import block after the lk_moe import, and one
#       `layer_done()` call at the end of `_cpu_decode`)
#   <tree>/sglang/srt/utils/dsv41_pool6row_probe.py          (implementation)
#   layer.py.orig-pool6row.{site,repo}                       (pre-edit backup)
#
# Usage:
#   apply_sglang_pool6row_probe.sh [--dry-run] [--status] [--rollback]
#                                  [--reinstall] [--tree site|repo|both]
#   exit 0 ok / 1 failure or --status "not applied" / 2 bad args
# =============================================================================
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/dsv41_pool6row_probe.py"
SITE_ROOT="/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages"
REPO_ROOT="/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python"
PYTHON="/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python"
REL="sglang/srt/layers/moe/fused_moe_triton/layer.py"
STEM="dsv41_pool6row_probe"
TAG="pool6row"

TREE="both"; MODE="apply"; REINSTALL=0
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --dry-run)  MODE="dryrun" ;;
    --status)   MODE="status" ;;
    --rollback) MODE="rollback" ;;
    --reinstall) REINSTALL=1 ;;
    --tree) TREE="${args[$((i+1))]:-}" ;;
    --tree=*) TREE="${args[$i]#--tree=}" ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown arg: ${args[$i]}" >&2; exit 2 ;;
  esac
done
case "$TREE" in site|repo|both) ;; *) echo "bad --tree: $TREE" >&2; exit 2 ;; esac

die()  { echo "[$TAG] ERROR: $*" >&2; exit 1; }
note() { printf '[%s] %s\n' "$TAG" "$*"; }
[[ -f "$SRC" ]] || die "missing $SRC"
[[ -x "$PYTHON" ]] || die "missing interpreter $PYTHON"

targets=()
[[ "$TREE" == "site" || "$TREE" == "both" ]] && targets+=("$SITE_ROOT")
[[ "$TREE" == "repo" || "$TREE" == "both" ]] && targets+=("$REPO_ROOT")

RC=0
for ROOT in "${targets[@]}"; do
  TGT="${ROOT}/${REL}"
  DEST="${ROOT}/sglang/srt/utils/${STEM}.py"
  SUFFIX=$( [[ "$ROOT" == *site-packages ]] && echo site || echo repo )
  BAK="${TGT}.orig-${TAG}.${SUFFIX}"
  if [[ ! -f "$TGT" ]]; then note "[$ROOT] target absent -> skip"; continue; fi
  n_imp=$(grep -c 'LK_POOL_6ROW_PROBE IMPORT' "$TGT" || true)
  n_call=$(grep -c 'LK_POOL_6ROW_PROBE CALL' "$TGT" || true)

  case "$MODE" in
    status)
      if (( n_imp > 0 && n_call > 0 )); then
        note "[$ROOT] APPLIED (import=$n_imp call=$n_call) backup=$BAK module=$([[ -f $DEST ]] && echo yes || echo NO)"
      else
        note "[$ROOT] NOT APPLIED (import=$n_imp call=$n_call)"
        RC=1
      fi
      continue ;;
    rollback)
      if [[ -f "$BAK" ]]; then
        cp -f "$BAK" "$TGT"; rm -f "$DEST"
        note "[$ROOT] ROLLED BACK from $BAK; removed $DEST"
      else
        note "[$ROOT] no backup $BAK -> nothing to do"
      fi
      continue ;;
    dryrun)
      note "[$ROOT] dry-run: would back up $TGT -> $BAK, splice 2 marker blocks, install $DEST, py_compile"
      continue ;;
  esac

  if (( n_imp > 0 && n_call > 0 )) && (( REINSTALL == 0 )); then
    note "[$ROOT] already applied (idempotent) -> refreshing module only"
    cp -f "$SRC" "$DEST"
    continue
  fi
  [[ -f "$BAK" ]] || cp -f "$TGT" "$BAK"
  cp -f "$SRC" "$DEST"
  "$PYTHON" - "$TGT" <<'PYEOF'
import io, re, sys
tgt = sys.argv[1]
src = io.open(tgt, encoding="utf-8").read()
A1 = ('logger.error("Failed to import lk_moe module or LK_MOE_HYBRID_ENABLED '
      'is not set to 1, lk::MOE implementation will not be available")')
A2 = '    def _cpu_decode(self, hidden_states, topk_weights, topk_ids):'
E1 = ('            self._pool_decode_launch(hidden_states, topk_ids, topk_weights,\n'
      '                                  real_tokens=real_tokens,\n'
      '                                  rows_per_token=rows_per_token)\n')
E2 = '        self.lk_moe.cpu_decode(\n            stream_ptr,\n            bs,\n'
E3 = '        output = FusedMoE.output_gpu[:hidden_states.size(0)]\n'
for name, needle in (("A1", A1), ("A2", A2), ("E1", E1), ("E2", E2), ("E3", E3)):
    n = src.count(needle)
    assert n == 1, "%s count=%d (expected 1)" % (name, n)
if 'LK_POOL_6ROW_PROBE IMPORT' in src:
    print("already marked; refusing to double-patch")
    sys.exit(3)
IMP = (
    '\n\n'
    '# >>> LK_POOL_6ROW_PROBE IMPORT (gate LK_POOL_6ROW_PROBE, default OFF) >>>\n'
    'import os as _os_6row\n'
    '_LK_POOL_6ROW_ON = str(_os_6row.environ.get("LK_POOL_6ROW_PROBE", "")).strip().lower() in ("1", "true", "on", "yes", "y")\n'
    '_lk_pool6row = None\n'
    'if _LK_POOL_6ROW_ON:\n'
    '    try:\n'
    '        from sglang.srt.utils.dsv41_pool6row_probe import probe as _p6row_probe\n'
    '\n'
    '        _lk_pool6row = _p6row_probe()\n'
    '    except Exception as _e_6row:  # never break serving for a probe\n'
    '        logger.warning("[pool6row] probe import failed: %r", _e_6row)\n'
    '        _lk_pool6row = None\n'
    '# <<< LK_POOL_6ROW_PROBE IMPORT <<<\n'
)
CALL = (
    '\n'
    '        # >>> LK_POOL_6ROW_PROBE CALL (default OFF; dsv41_pool6row_probe) >>>\n'
    '        _p6 = _lk_pool6row if _LK_POOL_6ROW_ON else None\n'
    '        # <<< LK_POOL_6ROW_PROBE CALL <<<\n'
)
DONE = (
    '        if _p6 is not None:\n'
    '            try:\n'
    '                _p6.layer_done(self, bs, rows_per_token, hidden_states,\n'
    '                               topk_ids, FusedMoE._pool_verify_rows_mult)\n'
    '            except Exception:\n'
    '                pass\n'
)
src = src.replace(A1, A1 + IMP, 1)
src = src.replace(A2, A2 + CALL, 1)
src = src.replace(E1, E1, 1)
src = src.replace(E2, E2, 1)
# ONE probe call site: the statement right before `_cpu_decode` returns its
# output, i.e. after the pool-stream join and the output_gpu accumulation, so
# the counters describe exactly what both halves were handed.
src = src.replace(E3, DONE + E3, 1)
io.open(tgt, "w", encoding="utf-8").write(src)
print("spliced 2 marker blocks + 2 layer_done call sites into %s" % tgt)
PYEOF
  "$PYTHON" -c "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True)" "$TGT" || die "py_compile failed: $TGT"
  "$PYTHON" -c "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True)" "$DEST" || die "py_compile failed: $DEST"
  note "[$ROOT] APPLIED (backup $BAK, module $DEST)"
done
exit "$RC"
