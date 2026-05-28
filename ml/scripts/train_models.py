from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"
VECTORSTORE_DIR = ROOT_DIR / "artifacts" / "vectorstore"


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def build_feature_frame(deliveries: pd.DataFrame, innings: int) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    subset = deliveries[(deliveries["innings"] == innings) & (deliveries["winner"] != "")].copy()
    if innings == 1:
        features = [
            "runs_before",
            "wickets_before",
            "legal_balls_before",
            "current_rr_before",
            "wickets_in_hand_before",
            "venue",
            "batting_team",
            "fielding_team",
        ]
        numeric_features = [
            "runs_before",
            "wickets_before",
            "legal_balls_before",
            "current_rr_before",
            "wickets_in_hand_before",
        ]
    else:
        features = [
            "runs_before",
            "wickets_before",
            "legal_balls_before",
            "current_rr_before",
            "wickets_in_hand_before",
            "target_score",
            "required_runs_before",
            "required_rr_before",
            "remaining_balls_before",
            "venue",
            "batting_team",
            "fielding_team",
        ]
        numeric_features = [
            "runs_before",
            "wickets_before",
            "legal_balls_before",
            "current_rr_before",
            "wickets_in_hand_before",
            "target_score",
            "required_runs_before",
            "required_rr_before",
            "remaining_balls_before",
        ]
    categorical_features = [column for column in features if column not in numeric_features]
    subset = subset[features + ["batting_team_won", "season_year"]].dropna()
    return subset[features], subset["batting_team_won"].astype(int), numeric_features, categorical_features


def make_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    base_model = LogisticRegression(max_iter=500, n_jobs=None)
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", calibrated),
        ]
    )


def evaluate_with_season_backtest(
    features: pd.DataFrame, target: pd.Series, seasons: pd.Series, numeric_features: list[str], categorical_features: list[str]
) -> dict:
    unique_seasons = sorted(seasons.unique().tolist())
    if len(unique_seasons) < 4:
        cutoff = unique_seasons[-1]
        train_mask = seasons < cutoff
        test_mask = seasons == cutoff
        holdouts = [cutoff]
    else:
        holdouts = unique_seasons[-3:]
        train_mask = seasons < holdouts[0]
        test_mask = seasons.isin(holdouts)

    model = make_model(numeric_features, categorical_features)
    model.fit(features[train_mask], target[train_mask])
    probabilities = model.predict_proba(features[test_mask])[:, 1]
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(features[train_mask], target[train_mask])
    baseline_probabilities = baseline.predict_proba(features[test_mask])[:, 1]

    return {
        "holdout_seasons": holdouts,
        "brier_score": round(float(brier_score_loss(target[test_mask], probabilities)), 5),
        "roc_auc": round(float(roc_auc_score(target[test_mask], probabilities)), 5),
        "baseline_brier_score": round(float(brier_score_loss(target[test_mask], baseline_probabilities)), 5),
        "model": model,
    }


def serialize_model(name: str, report: dict, numeric_features: list[str], categorical_features: list[str]) -> dict:
    model_path = MODELS_DIR / f"{name}.joblib"
    evaluation = {key: value for key, value in report.items() if key != "model"}
    joblib.dump(
        {
            "model": report["model"],
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "evaluation": evaluation,
            "model_version": "2026.05",
        },
        model_path,
    )
    return evaluation


def build_vectorstore(deliveries: pd.DataFrame, head_to_head: pd.DataFrame, pressure_batting: pd.DataFrame, pressure_bowling: pd.DataFrame) -> dict:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, str]] = []

    venue_phase = (
        deliveries.groupby(["venue", "phase"], as_index=False)
        .agg(avg_runs=("runs_total", "mean"), wickets=("wicket_fell", "sum"), balls=("legal_delivery", "sum"))
    )
    for row in venue_phase.itertuples(index=False):
        docs.append(
            {
                "id": f"venue:{row.venue}:{row.phase}",
                "title": f"{row.venue} {row.phase}",
                "text": f"At {row.venue} during the {row.phase}, deliveries average {row.avg_runs:.2f} runs and wickets fall every {max(row.balls / max(row.wickets, 1), 1):.1f} legal balls.",
            }
        )

    bowler_phase = (
        deliveries.groupby(["bowler", "phase"], as_index=False)
        .agg(runs=("runs_total", "sum"), balls=("legal_delivery", "sum"), wickets=("bowler_wicket", "sum"))
    )
    bowler_phase = bowler_phase[bowler_phase["balls"] >= 60]
    for row in bowler_phase.itertuples(index=False):
        economy = row.runs * 6 / max(row.balls, 1)
        strike = row.balls / max(row.wickets, 1)
        docs.append(
            {
                "id": f"bowler:{row.bowler}:{row.phase}",
                "title": f"{row.bowler} {row.phase} spell",
                "text": f"{row.bowler} concedes {economy:.2f} runs per over and takes a wicket every {strike:.1f} legal balls in the {row.phase}.",
            }
        )

    for row in head_to_head.head(250).itertuples(index=False):
        docs.append(
            {
                "id": f"h2h:{row.batter}:{row.bowler}",
                "title": f"{row.batter} vs {row.bowler}",
                "text": f"{row.batter} has scored {row.runs} runs from {row.balls} balls against {row.bowler} and been dismissed {row.dismissals} times.",
            }
        )

    for row in pressure_batting.head(75).itertuples(index=False):
        docs.append(
            {
                "id": f"pressure-bat:{row.player_name}",
                "title": f"{row.player_name} under pressure",
                "text": f"{row.player_name} scores at a strike rate of {row.strike_rate:.2f} across {row.balls} pressure balls with a pressure index of {row.pressure_index:.2f}.",
            }
        )

    for row in pressure_bowling.head(75).itertuples(index=False):
        docs.append(
            {
                "id": f"pressure-bowl:{row.player_name}",
                "title": f"{row.player_name} in pressure overs",
                "text": f"{row.player_name} has {row.wickets} wickets in {row.balls} pressure balls with an economy of {row.economy:.2f}.",
            }
        )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([doc["text"] for doc in docs])
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "documents": docs}, VECTORSTORE_DIR / "captain_vectorstore.joblib")
    return {"document_count": len(docs)}


def train_all() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    deliveries = load_processed("deliveries")
    head_to_head = load_processed("head_to_head")
    pressure_batting = load_processed("pressure_batting")
    pressure_bowling = load_processed("pressure_bowling")

    first_features, first_target, first_numeric, first_categorical = build_feature_frame(deliveries, innings=1)
    second_features, second_target, second_numeric, second_categorical = build_feature_frame(deliveries, innings=2)

    first_report = evaluate_with_season_backtest(
        first_features,
        first_target,
        deliveries.loc[first_features.index, "season_year"],
        first_numeric,
        first_categorical,
    )
    second_report = evaluate_with_season_backtest(
        second_features,
        second_target,
        deliveries.loc[second_features.index, "season_year"],
        second_numeric,
        second_categorical,
    )

    report_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "first_innings": serialize_model("first_innings_model", first_report, first_numeric, first_categorical),
        "second_innings": serialize_model("second_innings_model", second_report, second_numeric, second_categorical),
        "vectorstore": build_vectorstore(deliveries, head_to_head, pressure_batting, pressure_bowling),
    }
    (REPORTS_DIR / "model_report.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return report_payload


if __name__ == "__main__":
    train_all()

