import pytest
from fastapi.testclient import TestClient

from apps.api.app.main import app
from apps.api.app.services import captain


client = TestClient(app)


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch) -> None:
    # Keep API tests deterministic even when a local Ollama server is running.
    monkeypatch.setattr(captain, "_call_ollama", lambda prompt: None)


def test_dashboard_summary_endpoint() -> None:
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
    payload = response.json()
    assert "surprising_finding" in payload
    assert "team_toss_impact" in payload
    assert "team_innings_impact" in payload
    assert "team_milestone_impact" in payload
    assert len(payload["team_toss_impact"]) >= 10
    assert len(payload["team_innings_impact"]) >= 10
    assert len(payload["team_milestone_impact"]) >= 10
    assert len(payload["top_batters"]) == 5
    assert len(payload["top_bowlers"]) == 5


def test_predict_endpoint() -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "innings": 2,
            "current_score": 102,
            "wickets_lost": 3,
            "overs_completed": 12,
            "balls_into_over": 0,
            "target_score": 181,
            "batting_team": "Mumbai Indians",
            "bowling_team": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["win_probability"] <= 100
    assert payload["likely_winner"]


def test_options_include_team_player_pools() -> None:
    response = client.get("/api/v1/options")
    assert response.status_code == 200
    payload = response.json()
    assert "captain_batters_by_team" in payload
    assert "captain_bowlers_by_team" in payload
    assert len(payload["captain_batters_by_team"]) >= 10
    assert len(payload["captain_bowlers_by_team"]) >= 10
    assert all("team" in item and "players" in item for item in payload["captain_batters_by_team"][:3])
    assert all("team" in item and "players" in item for item in payload["captain_bowlers_by_team"][:3])


def test_captain_guardrail_for_non_phase_bowlers() -> None:
    response = client.post(
        "/api/v1/captain/advice",
        json={
            "innings": 2,
            "current_score": 148,
            "wickets_lost": 5,
            "overs_completed": 18,
            "balls_into_over": 0,
            "target_score": 191,
            "batting_team": "Chennai Super Kings",
            "bowling_team": "Mumbai Indians",
            "venue": "Wankhede Stadium",
            "batters": ["MS Dhoni", "RA Jadeja"],
            "available_bowlers": ["KA Pollard"],
            "remaining_overs_by_bowler": {"KA Pollard": 2},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is True
    assert "over_plan" in payload
    assert len(payload["over_plan"]) >= 1
    assert any(item["recommended_bowler"] is None for item in payload["over_plan"])
    assert "unassigned" in payload["answer"].lower() or "legal option" in payload["answer"].lower()


def test_captain_advice_supports_first_innings_bowling_context() -> None:
    response = client.post(
        "/api/v1/captain/advice",
        json={
            "innings": 1,
            "current_score": 54,
            "wickets_lost": 1,
            "overs_completed": 5,
            "balls_into_over": 2,
            "target_score": None,
            "batting_team": "Chennai Super Kings",
            "bowling_team": "Mumbai Indians",
            "venue": "Wankhede Stadium",
            "batters": ["RD Gaikwad", "DP Conway"],
            "available_bowlers": ["JJ Bumrah", "TA Boult", "HH Pandya"],
            "remaining_overs_by_bowler": {"JJ Bumrah": 2, "TA Boult": 2, "HH Pandya": 1},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert "over_plan" in payload
    assert 1 <= len(payload["over_plan"]) <= 5
    assert any(item["title"] == "Innings context" for item in payload["evidence"])


def test_captain_plan_respects_remaining_overs_and_blocks_consecutive_overs() -> None:
    response = client.post(
        "/api/v1/captain/advice",
        json={
            "innings": 2,
            "current_score": 142,
            "wickets_lost": 4,
            "overs_completed": 15,
            "balls_into_over": 0,
            "target_score": 188,
            "batting_team": "Chennai Super Kings",
            "bowling_team": "Mumbai Indians",
            "venue": "Wankhede Stadium",
            "batters": ["MS Dhoni", "RA Jadeja"],
            "available_bowlers": ["JJ Bumrah", "HH Pandya", "KA Pollard"],
            "remaining_overs_by_bowler": {"JJ Bumrah": 1, "HH Pandya": 3, "KA Pollard": 1},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assigned = [item for item in payload["over_plan"] if item["recommended_bowler"]]
    for previous, current in zip(assigned, assigned[1:]):
        assert previous["recommended_bowler"] != current["recommended_bowler"]
    counts: dict[str, int] = {}
    for item in assigned:
        counts[item["recommended_bowler"]] = counts.get(item["recommended_bowler"], 0) + 1
    assert counts.get("JJ Bumrah", 0) <= 1
    assert counts.get("KA Pollard", 0) <= 1


def test_insights_endpoint_returns_all_sections() -> None:
    response = client.get("/api/v1/insights")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["charts"]) == 4
    assert {row["model"] for row in payload["chase_models"]["leaderboard"]} == {
        "Logistic Regression",
        "Decision Tree",
        "SVM",
        "AdaBoost",
        "XGBoost",
    }
    assert {row["method"] for row in payload["chase_models"]["tuning"]} == {"GridSearchCV", "RandomizedSearchCV"}
    assert len(payload["freak_innings"]) == 10
    assert len(payload["batter_styles"]["clusters"]) == 4
    coach = payload["chase_coach"]
    assert len(coach["policy"]) == len(coach["balls_buckets"]) * len(coach["rate_buckets"]) * len(coach["wickets_buckets"])
    assert coach["balls_buckets"][-1] == "115-120"


def test_insight_chart_endpoint_serves_png_and_rejects_unknown_names() -> None:
    response = client.get("/api/v1/insights/charts/runs_by_phase")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert client.get("/api/v1/insights/charts/not_a_chart").status_code == 404
    assert client.get("/api/v1/insights/charts/decision_tree").status_code == 200


def test_predict_adds_score_projection_for_first_innings() -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "innings": 1,
            "current_score": 90,
            "wickets_lost": 2,
            "overs_completed": 10,
            "balls_into_over": 0,
            "batting_team": "Mumbai Indians",
            "bowling_team": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["projected_score"]["linear_regression"] >= 90
    assert payload["projected_score"]["svr"] >= 90
    assert payload["model_opinions"] is None


def test_predict_adds_model_opinions_for_chase() -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "innings": 2,
            "current_score": 120,
            "wickets_lost": 4,
            "overs_completed": 15,
            "balls_into_over": 0,
            "target_score": 175,
            "batting_team": "Mumbai Indians",
            "bowling_team": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert {item["model"] for item in payload["model_opinions"]} == {"Decision Tree", "SVM", "AdaBoost", "XGBoost"}
    assert all(0 <= item["probability"] <= 100 for item in payload["model_opinions"])
    assert len(payload["decision_path"]) >= 2
    assert payload["projected_score"] is None


def test_ollama_model_resolution_uses_installed_models_only(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "qwen2.5:3b"}, {"name": "llama3.2:1b"}]}

    monkeypatch.setattr(captain, "_resolved_ollama_model", None)
    monkeypatch.setattr(captain.requests, "get", lambda *args, **kwargs: FakeResponse())
    assert captain._resolve_ollama_model() == "llama3.2:1b"

    def unreachable(*args, **kwargs):
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(captain, "_resolved_ollama_model", None)
    monkeypatch.setattr(captain.requests, "get", unreachable)
    assert captain._resolve_ollama_model() is None


def test_ollama_bind_address_is_converted_to_client_url() -> None:
    from apps.api.app.config import _client_ollama_host

    assert _client_ollama_host("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert _client_ollama_host("http://localhost:11434/") == "http://localhost:11434"


def test_captain_plan_uses_every_quota_spin_first_and_pace_last() -> None:
    bowlers = ["PP Ojha", "RP Singh", "A Mishra", "A Symonds", "DW Steyn"]
    response = client.post(
        "/api/v1/captain/advice",
        json={
            "innings": 2,
            "current_score": 132,
            "wickets_lost": 4,
            "overs_completed": 15,
            "balls_into_over": 0,
            "target_score": 185,
            "batting_team": "Chennai Super Kings",
            "bowling_team": "Deccan Chargers",
            "venue": "Wankhede Stadium",
            "batters": ["MS Dhoni", "SK Raina"],
            "available_bowlers": bowlers,
            "remaining_overs_by_bowler": {"PP Ojha": 1, "RP Singh": 1, "A Mishra": 1, "A Symonds": 2, "DW Steyn": 0},
        },
    )
    assert response.status_code == 200
    plan = response.json()["over_plan"]
    assigned = [item["recommended_bowler"] for item in plan]
    assert None not in assigned
    assert assigned.count("A Symonds") == 2
    assert assigned[-1] == "RP Singh"
    assert all(item["bowler_type"] for item in plan)
