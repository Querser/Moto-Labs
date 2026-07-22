#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT"

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" != "arm64" ]; then
    echo "Moto Laps supports Apple Silicon Macs (M1 or newer); Intel Macs are not supported." >&2
    exit 1
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    PYTHON=""
    for candidate in python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
            PYTHON=$candidate
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Python 3.10+ was not found. Install Python and run this script again." >&2
        exit 1
    fi
    echo "Creating local virtual environment..."
    "$PYTHON" -m venv .venv
fi

if ! "$VENV_PYTHON" -c 'import fastapi, cv2, sqlalchemy, openpyxl, rapidocr_onnxruntime, onnxruntime, supervision, paddleocr, torch, transformers' >/dev/null 2>&1; then
    echo "Installing project dependencies into .venv..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -e .
fi

PPOCR_MODEL="$PROJECT_ROOT/models/paddlex/official_models/PP-OCRv6_medium_rec_onnx/inference.onnx"
PPOCR_DETECTOR="$PROJECT_ROOT/models/paddlex/official_models/PP-OCRv5_mobile_det_onnx/inference.onnx"
FLORENCE_MODEL="$PROJECT_ROOT/models/florence-2-base-ft/model.safetensors"
if [ ! -f "$PPOCR_MODEL" ] || [ ! -f "$PPOCR_DETECTOR" ] || [ ! -f "$FLORENCE_MODEL" ]; then
    echo "Downloading the pinned local OCR models..."
    "$VENV_PYTHON" scripts/setup_models.py
fi

MODEL_PATH="$PROJECT_ROOT/models/yolox_tiny.onnx"
MODEL_SHA256="427CC366D34E27FF7A03E2899B5E3671425C262EA2291F88BB942BC1CC70B0F7"
MODEL_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    mkdir -p "$PROJECT_ROOT/models"
    echo "Downloading the official YOLOX-Tiny detector model..."
    MODEL_PATH="$MODEL_PATH" MODEL_URL="$MODEL_URL" "$VENV_PYTHON" -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ["MODEL_URL"], os.environ["MODEL_PATH"] + ".download")'
    mv "$MODEL_PATH.download" "$MODEL_PATH"
fi
ACTUAL_MODEL_SHA256=$(MODEL_PATH="$MODEL_PATH" "$VENV_PYTHON" -c 'import hashlib, os, pathlib; print(hashlib.sha256(pathlib.Path(os.environ["MODEL_PATH"]).read_bytes()).hexdigest().upper())')
if [ "$ACTUAL_MODEL_SHA256" != "$MODEL_SHA256" ]; then
    echo "Detector model checksum is invalid: $MODEL_PATH" >&2
    exit 1
fi

if [ -f "$PROJECT_ROOT/alembic.ini" ]; then
    "$VENV_PYTHON" -m alembic upgrade head
fi

echo "Opening Moto Laps at http://127.0.0.1:8000"
exec "$VENV_PYTHON" -m app.cli "$@"
