#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$DIR/sglang-dsv41-autotune-dummy-decode.patch"
ENV_PREFIX="${ENV_PREFIX:-/mnt/SSD01/models/.llm-env/Lsglang/env}"
SITE="${SITE:-$ENV_PREFIX/lib/python3.12/site-packages}"
F="$SITE/sglang/srt/model_executor/runner/base_runner.py"
MARK="SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE"
mkdir -p "$DIR"
[[ -f "$PATCH" && -f "$F" ]] || { echo "ERROR: missing patch or target" >&2; exit 1; }
if grep -q "$MARK" "$F"; then echo "ALREADY APPLIED; nothing to do."; exit 0; fi
if [[ -f "$F.orig3" ]]; then echo "WARNING: $F.orig3 exists; run rollback first." >&2; exit 1; fi
cp -p "$F" "$F.orig3"; echo "backup : $F.orig3"
if patch -p1 --dry-run -d "$SITE" < "$PATCH"; then
  patch -p1 -d "$SITE" < "$PATCH"; echo "APPLIED."
else
  echo "ERROR: patch failed; restoring." >&2; cp -p "$F.orig3" "$F"; exit 1
fi
"$ENV_PREFIX/bin/python" -m py_compile "$F" && echo "py_compile OK"
