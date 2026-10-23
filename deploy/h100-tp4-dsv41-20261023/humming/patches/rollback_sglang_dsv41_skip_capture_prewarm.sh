#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F="$SITE/sglang/srt/models/deepseek_v4.py"
MARK="SGLANG_DSV41_SKIP_CAPTURE_PREWARM"
[[ -f "$F.orig2" ]] || { echo "Nothing to roll back ($F.orig2 absent)."; exit 0; }
cp -p "$F.orig2" "$F"; rm -f "$F.orig2"
grep -q "$MARK" "$F" && { echo "WARNING: marker still present." >&2; exit 1; }
echo "ROLLED BACK: $F restored (③'s scope patch preserved)."
