from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

from .config import Settings, get_settings


class AnalyticsRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connection = duckdb.connect(database=":memory:")
        self._register_views()

    def _register_views(self) -> None:
        parquet_files = {
            "deliveries": self.settings.processed_dir / "deliveries.parquet",
            "matches": self.settings.processed_dir / "matches.parquet",
            "head_to_head": self.settings.processed_dir / "head_to_head.parquet",
            "partnerships": self.settings.processed_dir / "partnerships.parquet",
            "pressure_batting": self.settings.processed_dir / "pressure_batting.parquet",
            "pressure_bowling": self.settings.processed_dir / "pressure_bowling.parquet",
            "player_metadata": self.settings.processed_dir / "player_metadata.parquet",
        }
        for view_name, path in parquet_files.items():
            if path.exists():
                self.connection.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path.as_posix()}')")

    def read_json_report(self, name: str) -> dict[str, Any]:
        path = self.settings.reports_dir / name
        return json.loads(path.read_text(encoding="utf-8"))

    def read_model_bundle(self, name: str) -> dict[str, Any]:
        return joblib.load(self.settings.models_dir / f"{name}.joblib")

    def read_vectorstore(self) -> dict[str, Any]:
        return joblib.load(self.settings.vectorstore_dir / "captain_vectorstore.joblib")

    def chart_path(self, name: str) -> Path:
        return self.settings.reports_dir / "charts" / f"{name}.png"

    def fetch_dataframe(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.connection.execute(sql, params or []).df()


@lru_cache
def get_repository() -> AnalyticsRepository:
    return AnalyticsRepository(get_settings())

