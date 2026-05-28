from __future__ import annotations

import math

import pandas as pd

from ..data_access import AnalyticsRepository
from ..models import WinPredictRequest, WinPredictResponse


def _request_to_frame(payload: WinPredictRequest) -> pd.DataFrame:
    legal_balls_before = payload.overs_completed * 6 + payload.balls_into_over
    current_rr_before = payload.current_score * 6 / legal_balls_before if legal_balls_before > 0 else 0.0
    remaining_balls_before = max(120 - legal_balls_before, 0)
    target_score = payload.target_score
    required_runs_before = max((target_score or 0) - payload.current_score, 0) if payload.innings == 2 else None
    required_rr_before = (
        required_runs_before * 6 / remaining_balls_before if payload.innings == 2 and remaining_balls_before > 0 else None
    )
    values = {
        "runs_before": payload.current_score,
        "wickets_before": payload.wickets_lost,
        "legal_balls_before": legal_balls_before,
        "current_rr_before": current_rr_before,
        "wickets_in_hand_before": 10 - payload.wickets_lost,
        "venue": payload.venue,
        "batting_team": payload.batting_team,
        "fielding_team": payload.bowling_team,
        "target_score": target_score,
        "required_runs_before": required_runs_before,
        "required_rr_before": required_rr_before,
        "remaining_balls_before": remaining_balls_before,
    }
    return pd.DataFrame([values])


def _top_factors(payload: WinPredictRequest, probability: float) -> list[str]:
    legal_balls_before = payload.overs_completed * 6 + payload.balls_into_over
    factors: list[str] = []
    if payload.innings == 2 and payload.target_score:
        required = max(payload.target_score - payload.current_score, 0)
        remaining = max(120 - legal_balls_before, 1)
        required_rr = required * 6 / remaining
        factors.append(f"Required run rate is {required_rr:.2f} with {required} still needed.")
    else:
        current_rr = payload.current_score * 6 / legal_balls_before if legal_balls_before > 0 else 0.0
        factors.append(f"Current scoring rate is {current_rr:.2f} with {10 - payload.wickets_lost} wickets still in hand.")
    factors.append(f"The model leans {'positive' if probability >= 0.5 else 'negative'} because {payload.bowling_team} must respond at {payload.venue}.")
    if payload.wickets_lost >= 6:
        factors.append("A thin batting tail is sharply reducing the upside from here.")
    elif payload.wickets_lost <= 2:
        factors.append("A strong platform is preserving acceleration potential.")
    return factors[:3]


def predict(repository: AnalyticsRepository, payload: WinPredictRequest) -> WinPredictResponse:
    bundle_name = "first_innings_model" if payload.innings == 1 else "second_innings_model"
    bundle = repository.read_model_bundle(bundle_name)
    frame = _request_to_frame(payload)
    expected_columns = bundle["numeric_features"] + bundle["categorical_features"]
    frame = frame[expected_columns]
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    confidence_width = 0.08 if 0.35 <= probability <= 0.65 else 0.05
    lower = max(probability - confidence_width, 0.0)
    upper = min(probability + confidence_width, 1.0)
    likely_winner = payload.batting_team if probability >= 0.5 else payload.bowling_team
    evaluation = bundle["evaluation"]
    stamp = f"ROC-AUC {evaluation['roc_auc']}, Brier {evaluation['brier_score']}"
    return WinPredictResponse(
        win_probability=round(probability * 100, 2),
        confidence_band=(round(lower * 100, 2), round(upper * 100, 2)),
        likely_winner=likely_winner,
        top_factors=_top_factors(payload, probability),
        model_version=bundle["model_version"],
        evaluation_stamp=stamp,
    )

