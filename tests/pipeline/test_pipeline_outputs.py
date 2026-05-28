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
