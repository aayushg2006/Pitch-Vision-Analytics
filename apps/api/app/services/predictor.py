from __future__ import annotations

import math

import pandas as pd

from ..data_access import AnalyticsRepository
from ..models import ModelOpinion, ProjectedScore, WinPredictRequest, WinPredictResponse


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


def _projected_score(repository: AnalyticsRepository, payload: WinPredictRequest, frame: pd.DataFrame) -> ProjectedScore | None:
    try:
        bundle = repository.read_model_bundle("score_models")
    except FileNotFoundError:
        return None
    features = frame[bundle["features"]]
    simple = bundle["simple"]
    return ProjectedScore(
        linear_regression=round(max(float(bundle["multiple"].predict(features)[0]), payload.current_score), 1),
        svr=round(max(float(bundle["svr"].predict(features)[0]), payload.current_score), 1),
        rule_of_thumb=f"Final score ≈ {simple.intercept_:.0f} + {simple.coef_[0]:.2f} × runs at 10 overs",
    )


def _trim(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _describe_tree_path(tree, features: pd.DataFrame, labels: list[str]) -> list[str]:
    nodes = tree.decision_path(features).indices
    leaf = int(tree.apply(features)[0])
    steps: list[str] = []
    for node in nodes:
        if node == leaf:
            continue
        index = tree.tree_.feature[node]
        threshold = tree.tree_.threshold[node]
        value = float(features.iloc[0, index])
        sign = "≤" if value <= threshold else ">"
        decimals = next((places for places in (2, 3, 4) if _trim(value, places) != _trim(threshold, places)), 4)
        steps.append(f"{labels[index]} is {_trim(value, decimals)} ({sign} {_trim(threshold, decimals)})")
    chase_rate = float(tree.predict_proba(features)[0, 1]) * 100
    steps.append(f"Chasing sides in this situation won {chase_rate:.0f}% of the time.")
    return steps


def _chase_opinions(
    repository: AnalyticsRepository, frame: pd.DataFrame
) -> tuple[list[ModelOpinion] | None, list[str] | None]:
    row = frame.iloc[0]
    if not row["required_runs_before"] or not row["remaining_balls_before"]:
        return None, None
    try:
        bundle = repository.read_model_bundle("chase_models")
    except FileNotFoundError:
        return None, None
    features = frame[bundle["features"]].astype(float)
    opinions = [
        ModelOpinion(model=name, probability=round(float(model.predict_proba(features)[0, 1]) * 100, 2))
        for name, model in bundle["models"].items()
        if name != "Logistic Regression"
    ]
    labels = [bundle["feature_labels"][name] for name in bundle["features"]]
    return opinions, _describe_tree_path(bundle["models"]["Decision Tree"], features, labels)


def predict(repository: AnalyticsRepository, payload: WinPredictRequest) -> WinPredictResponse:
    bundle_name = "first_innings_model" if payload.innings == 1 else "second_innings_model"
    bundle = repository.read_model_bundle(bundle_name)
    full_frame = _request_to_frame(payload)
    expected_columns = bundle["numeric_features"] + bundle["categorical_features"]
    frame = full_frame[expected_columns]
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    projected_score = _projected_score(repository, payload, full_frame) if payload.innings == 1 else None
    model_opinions, decision_path = _chase_opinions(repository, full_frame) if payload.innings == 2 else (None, None)
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
        projected_score=projected_score,
        model_opinions=model_opinions,
        decision_path=decision_path,
    )

