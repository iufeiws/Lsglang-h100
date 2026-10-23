#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F="$SITE/sglang/srt/layers/attention/deepseek_v4_backend.py"
MARK="SGLANG_DSV41_DUMP_FLASHMLA_ARGS"
[[ -f "$F.orig5" ]] || { echo "Nothing to roll back ($F.orig5 absent)."; exit 0; }
cp -p "$F.orig5" "$F"; rm -f "$F.orig5"
grep -q "$MARK" "$F" && { echo "WARNING: marker still present." >&2; exit 1; }
echo "ROLLED BACK: $F restored (dump removed)."
