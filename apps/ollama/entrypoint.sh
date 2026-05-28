#!/bin/sh
set -eu

export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"

ollama serve &
pid=$!

sleep 5
curl -fsS http://127.0.0.1:11434/api/generate -d '{"model":"tinyllama","prompt":"","stream":false,"keep_alive":-1}' >/dev/null 2>&1 || true

wait "$pid"
