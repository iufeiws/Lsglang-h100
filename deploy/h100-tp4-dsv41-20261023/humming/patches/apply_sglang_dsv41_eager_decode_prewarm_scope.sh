#!/usr/bin/env bash
# Apply the DSV4.1 MoE eager-decode-prewarm scope patch (BACKUP for fix ③).
# Touches TWO files inside site-packages and keeps .orig backups next to them.
# Does NOT touch any repo working tree.
#
# Usage:
#   bash apply_sglang_dsv41_eager_decode_prewarm_scope.sh
#   SITE=/path/to/site-packages bash apply_sglang_dsv41_eager_decode_prewarm_scope.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-dsv41-moe-eager-decode-prewarm-scope.patch"
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"

F1="$SITE/sglang/srt/layers/moe/fused_moe_triton/layer.py"
F2="$SITE/sglang/srt/models/deepseek_v4.py"
MARK="_eager_decode_prewarm"

for f in "$PATCH" "$F1" "$F2"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

echo "site   : $SITE"
echo "patch  : $PATCH"

if grep -q "$MARK" "$F1" && grep -q "$MARK" "$F2"; then
  echo "ALREADY APPLIED (marker present in both files); nothing to do."
  exit 0
fi

for f in "$F1" "$F2"; do
  if [[ -f "$f.orig" ]]; then
    echo "WARNING: $f.orig already exists; refusing to overwrite." >&2
    echo "         run the rollback script first for a clean re-apply." >&2
    exit 1
  fi
done

for f in "$F1" "$F2"; do cp -p "$f" "$f.orig"; echo "backup : $f.orig"; done

if patch -p1 --dry-run -d "$SITE" < "$PATCH"; then
  patch -p1 -d "$SITE" < "$PATCH"
  echo "APPLIED."
else
  echo "ERROR: patch did not apply cleanly; restoring." >&2
  for f in "$F1" "$F2"; do [[ -f "$f.orig" ]] && cp -p "$f.orig" "$f"; done
  exit 1
fi

echo
echo "Sanity:"
echo "  grep -n $MARK $F1 $F2"
echo "  $ENV_PREFIX/bin/python -m py_compile $F1 $F2"
