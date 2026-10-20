import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_outputs_exist() -> None:
    assert (ROOT / "data" / "processed" / "deliveries.parquet").exists()
    assert (ROOT / "artifacts" / "models" / "first_innings_model.joblib").exists()
    assert (ROOT / "artifacts" / "vectorstore" / "captain_vectorstore.joblib").exists()


def test_audit_summary_shape() -> None:
    payload = json.loads((ROOT / "artifacts" / "reports" / "data_audit_summary.json").read_text(encoding="utf-8"))
    assert payload["rows_in_raw"] > 0
    assert "cleaning_script" in payload
    assert "derived_fields" in payload


def test_ml_insight_outputs_exist() -> None:
    assert (ROOT / "artifacts" / "models" / "score_models.joblib").exists()
    assert (ROOT / "artifacts" / "models" / "chase_models.joblib").exists()
    for chart in ["runs_by_phase", "toss_decision_wins", "feature_correlation", "season_scoring_trend", "decision_tree"]:
        assert (ROOT / "artifacts" / "reports" / "charts" / f"{chart}.png").exists()
    payload = json.loads((ROOT / "artifacts" / "reports" / "model_insights.json").read_text(encoding="utf-8"))
    for key in ["charts", "score_models", "chase_models", "freak_innings", "batter_styles", "chase_coach"]:
        assert key in payload
