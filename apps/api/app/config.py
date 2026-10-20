from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    models_dir: Path
    vectorstore_dir: Path
    ollama_host: str
    ollama_model: str


def _client_ollama_host(raw: str) -> str:
    # OLLAMA_HOST is also Ollama's own bind setting (e.g. "0.0.0.0:11434"), so turn it into a usable client URL.
    host = raw if "://" in raw else f"http://{raw}"
    return host.replace("://0.0.0.0", "://127.0.0.1").rstrip("/")


@lru_cache
def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("APP_DATA_DIR", root_dir / "data"))
    artifacts_dir = Path(os.getenv("APP_ARTIFACTS_DIR", root_dir / "artifacts"))
    return Settings(
        root_dir=root_dir,
        data_dir=data_dir,
        processed_dir=data_dir / "processed",
        artifacts_dir=artifacts_dir,
        reports_dir=artifacts_dir / "reports",
        models_dir=artifacts_dir / "models",
        vectorstore_dir=artifacts_dir / "vectorstore",
        ollama_host=_client_ollama_host(os.getenv("OLLAMA_HOST", "http://localhost:11434")),
        ollama_model=os.getenv("OLLAMA_MODEL", "auto"),
    )

