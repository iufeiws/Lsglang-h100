#!/usr/bin/env bash
# 回滚 sglang-dsv41-accept-trace (= apply_sglang_dsv41_accept_trace.sh --rollback --tree both)
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/apply_sglang_dsv41_accept_trace.sh" --rollback --tree both
