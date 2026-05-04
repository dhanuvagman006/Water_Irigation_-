#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "========================================="
echo "Starting Water Irrigation System"
echo "========================================="
echo "Press Ctrl+C to stop both servers."
echo

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	echo "[!] Python not found on PATH (expected python or python3)."
	exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
	echo "[!] npm not found on PATH. Please install Node.js."
	exit 1
fi

VENV_DIR="$ROOT_DIR/.venv"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
	# shellcheck disable=SC1091
	source "$VENV_DIR/Scripts/activate"
else
	echo "[!] Virtual environment not found at: $VENV_DIR"
	echo "    Create it with: python -m venv .venv"
	echo "    Then re-run this script."
	exit 1
fi

echo "[1/3] Activating Virtual Environment and Training Model..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$ROOT_DIR/backend/requirements.txt"
# python "$ROOT_DIR/backend/app/ml/train.py"

echo "[2/3] Starting Backend Server on port 8001..."
backend_pid=""
cleanup() {
	if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
		echo
		echo "Stopping backend (PID $backend_pid)..."
		kill "$backend_pid" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload &
backend_pid=$!
cd "$ROOT_DIR"

echo "[3/3] Installing Dependencies and Starting Frontend Server..."
cd "$ROOT_DIR/frontend"
npm install
npm run dev -- --host 127.0.0.1
