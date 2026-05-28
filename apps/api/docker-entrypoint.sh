#!/bin/sh
set -eu

cd /workspace

ARTIFACT_FLAG="artifacts/models/first_innings_model.joblib"

if [ ! -f "$ARTIFACT_FLAG" ]; then
  echo "Processed artifacts not found. Building pipeline and models first..."
  python ml/scripts/build_all.py
fi

echo "Starting FastAPI backend..."
exec uvicorn apps.api.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
