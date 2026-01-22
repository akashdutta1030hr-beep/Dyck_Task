#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to generate the Dyck datasets.
# Uses PYTHON env var if provided, otherwise tries python3 then python.

PY_BIN="${PYTHON:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=python
fi

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Error: Python is not installed or not on PATH." >&2
  exit 1
fi

echo "Running dataset generation with ${PY_BIN}..."
"$PY_BIN" generator.py
