#!/usr/bin/env bash
# One-click rollback of the DSV4.1 MoE eager-decode-prewarm scope patch.
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F1="$SITE/sglang/srt/layers/moe/fused_moe_triton/layer.py"
F2="$SITE/sglang/srt/models/deepseek_v4.py"
MARK="_eager_decode_prewarm"

restored=0
for f in "$F1" "$F2"; do
  if [[ -f "$f.orig" ]]; then
    cp -p "$f.orig" "$f"; rm -f "$f.orig"
    echo "RESTORED: $f"
    restored=1
  fi
done

if [[ "$restored" == "0" ]]; then
  echo "Nothing to roll back (no .orig backups found)."
  exit 0
fi

if grep -q "$MARK" "$F1" || grep -q "$MARK" "$F2"; then
  echo "WARNING: marker still present after restore; inspect manually." >&2
  exit 1
fi
echo "ROLLED BACK (no '$MARK' anywhere; upstream bodies restored)."
