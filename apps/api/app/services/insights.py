from __future__ import annotations

from pathlib import Path

from ..data_access import AnalyticsRepository
from ..models import InsightsResponse


CHART_NAMES = {"runs_by_phase", "toss_decision_wins", "feature_correlation", "season_scoring_trend", "decision_tree"}


def get_insights(repository: AnalyticsRepository) -> InsightsResponse:
    return InsightsResponse.model_validate(repository.read_json_report("model_insights.json"))


def get_chart_path(repository: AnalyticsRepository, name: str) -> Path | None:
    if name not in CHART_NAMES:
        return None
    path = repository.chart_path(name)
    return path if path.exists() else None
