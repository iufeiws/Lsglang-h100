#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F="$SITE/sglang/srt/model_executor/runner/base_runner.py"
MARK="SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE"
[[ -f "$F.orig3" ]] || { echo "Nothing to roll back ($F.orig3 absent)."; exit 0; }
cp -p "$F.orig3" "$F"; rm -f "$F.orig3"
grep -q "$MARK" "$F" && { echo "WARNING: marker still present." >&2; exit 1; }
echo "ROLLED BACK: only this patch undone (③ and SKIP patches preserved)."
