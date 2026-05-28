from fastapi.testclient import TestClient

from apps.api.app.main import app


client = TestClient(app)


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
