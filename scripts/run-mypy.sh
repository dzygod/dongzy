#!/usr/bin/env bash
# pre-commit / CI：在 Apple Silicon 的 Rosetta 终端里，.venv 多为 arm64 轮子；
# 若用 x86_64 解释器加载会报 incompatible architecture，此处强制原生 arm64。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "未找到 .venv，请先: python -m venv .venv && pip install -e '.[dev]'" >&2
  exit 1
fi
if [[ "$(uname -s)" == "Darwin" && "$(arch)" == "i386" ]]; then
  # Apple Silicon 在 Rosetta 终端：hw.optional.arm64=1；真 Intel Mac 为 0
  if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    exec arch -arm64 "$PY" -m mypy src
  fi
fi
exec "$PY" -m mypy src
