from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import get_settings
from ..data_access import AnalyticsRepository
from ..models import (
    CaptainAdviceRequest,
    CaptainPlanOver,
    CaptainResponse,
    EvidenceItem,
    ImpactOptimizationRequest,
    ImpactOptimizationResponse,
    ImpactScenario,
)
from .predictor import predict


ROLE_ADJUSTMENTS = {
    "finisher": {"late": 7.0, "middle": 4.0, "early": 1.5},
    "anchor": {"late": 2.0, "middle": 5.0, "early": 6.0},
    "spinner": {"late": 2.0, "middle": 5.0, "early": 3.0},
    "pacer": {"late": 6.0, "middle": 3.0, "early": 5.0},
    "all_rounder": {"late": 4.0, "middle": 4.0, "early": 4.0},
}

PHASE_MIN_INNINGS = 10
DEBUT_TOTAL_MATCHES_LIMIT = 15
DEBUT_PHASE_MIN_INNINGS = 3
PLAN_OVERS_HORIZON = 5
MAX_OVERS_PER_BOWLER = 4


def _retrieve_context(repository: AnalyticsRepository, prompt: str, limit: int = 5) -> list[dict[str, Any]]:
    vectorstore = repository.read_vectorstore()
    vector = vectorstore["vectorizer"].transform([prompt])
    scores = (vectorstore["matrix"] @ vector.T).toarray().ravel()
    indices = np.argsort(scores)[::-1][:limit]
    documents = []
    for index in indices:
        if scores[index] <= 0:
            continue
        document = vectorstore["documents"][int(index)]
        documents.append({"score": float(scores[index]), **document})
    return documents


def _call_ollama(prompt: str) -> str | None:
    settings = get_settings()
    try:
        response = requests.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()
    except Exception:
        return None


def _current_phase(payload: CaptainAdviceRequest) -> str:
    if payload.overs_completed < 6:
        return "powerplay"
    if payload.overs_completed < 15:
        return "middle"
    return "death"


def _phase_label(phase: str) -> str:
    return {
        "powerplay": "powerplay",
        "middle": "middle overs",
        "death": "death overs",
    }[phase]


def _innings_context_label(payload: CaptainAdviceRequest) -> str:
    if payload.innings == 1:
        return "first-innings bowling"
    return "target defense"


def _phase_for_over_number(over_number: int) -> str:
    if over_number <= 6:
        return "powerplay"
    if over_number <= 15:
        return "middle"
    return "death"


def _next_over_numbers(payload: CaptainAdviceRequest, horizon: int = PLAN_OVERS_HORIZON) -> list[int]:
    start_over = payload.overs_completed + 1
    if payload.balls_into_over > 0:
        start_over += 1
    start_over = max(1, min(start_over, 20))
    end_over = min(20, start_over + horizon - 1)
    return list(range(start_over, end_over + 1))


def _normalize_remaining_overs(payload: CaptainAdviceRequest) -> dict[str, int]:
    normalized: dict[str, int] = {}
    selected = set(payload.available_bowlers)
    for bowler in payload.available_bowlers:
        raw_value = payload.remaining_overs_by_bowler.get(bowler, 0)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 0
        normalized[bowler] = max(0, min(MAX_OVERS_PER_BOWLER, value))
    for bowler in list(payload.remaining_overs_by_bowler.keys()):
        if bowler not in selected:
            continue
    return normalized


def _placeholder_list(values: list[str]) -> str:
    return ", ".join(["?"] * len(values))


def _build_candidate_frame(repository: AnalyticsRepository, payload: CaptainAdviceRequest, phase: str) -> pd.DataFrame:
    if not payload.available_bowlers:
        return pd.DataFrame()

    phase_rows = repository.fetch_dataframe(
        f"""
        SELECT
            bowler,
            COUNT(DISTINCT match_id) AS phase_innings,
            SUM(runs_total) AS runs,
            SUM(legal_delivery) AS balls,
            SUM(bowler_wicket) AS wickets
        FROM deliveries
        WHERE fielding_team = ?
          AND phase = ?
          AND bowler IN ({_placeholder_list(payload.available_bowlers)})
        GROUP BY bowler
        """,
        [payload.bowling_team, phase, *payload.available_bowlers],
    )
    if phase_rows.empty:
        return phase_rows

    total_rows = repository.fetch_dataframe(
        f"""
        SELECT bowler, COUNT(DISTINCT match_id) AS total_matches
        FROM deliveries
        WHERE bowler IN ({_placeholder_list(payload.available_bowlers)})
        GROUP BY bowler
        """,
        payload.available_bowlers,
    )
    matchups = repository.fetch_dataframe(
        f"""
        SELECT
            bowler,
            SUM(runs) AS matchup_runs,
            SUM(balls) AS matchup_balls,
            SUM(dismissals) AS matchup_dismissals
        FROM head_to_head
        WHERE batter IN ({_placeholder_list(payload.batters or [''])})
          AND bowler IN ({_placeholder_list(payload.available_bowlers)})
        GROUP BY bowler
        """,
        [*(payload.batters or [""]), *payload.available_bowlers],
    )

    candidate_frame = phase_rows.merge(total_rows, on="bowler", how="left").merge(matchups, on="bowler", how="left")
    candidate_frame[["matchup_runs", "matchup_balls", "matchup_dismissals"]] = candidate_frame[
        ["matchup_runs", "matchup_balls", "matchup_dismissals"]
    ].fillna(0)
    candidate_frame["economy"] = candidate_frame["runs"] * 6 / candidate_frame["balls"].clip(lower=1)
    candidate_frame["strike_rate_bowling"] = candidate_frame["balls"] / candidate_frame["wickets"].clip(lower=1)
    candidate_frame["matchup_strike_rate"] = np.where(
        candidate_frame["matchup_balls"] > 0,
        candidate_frame["matchup_runs"] * 100 / candidate_frame["matchup_balls"],
        130.0,
    )
    candidate_frame["eligible_standard"] = candidate_frame["phase_innings"] >= PHASE_MIN_INNINGS
    candidate_frame["eligible_debut"] = (
        (candidate_frame["total_matches"] < DEBUT_TOTAL_MATCHES_LIMIT)
        & (candidate_frame["phase_innings"] >= DEBUT_PHASE_MIN_INNINGS)
    )
    candidate_frame["eligible"] = candidate_frame["eligible_standard"] | candidate_frame["eligible_debut"]
    phase_weight = {"powerplay": 1.0, "middle": 0.9, "death": 1.12}[phase]
    innings_weight = 1.15 if payload.innings == 1 else 1.0
    candidate_frame["score"] = (
        candidate_frame["wickets"] * 2.8 * phase_weight * innings_weight
        - candidate_frame["economy"] * 1.5
        - candidate_frame["strike_rate_bowling"] * 0.08
        + candidate_frame["phase_innings"] * 0.18
        + candidate_frame["matchup_dismissals"] * 2.25
        - candidate_frame["matchup_strike_rate"] * 0.02
    )
    return candidate_frame.sort_values(["eligible", "score", "phase_innings"], ascending=[False, False, False]).reset_index(drop=True)


def _build_manual_answer(
    over_plan: list[CaptainPlanOver],
    payload: CaptainAdviceRequest,
) -> str:
    if not over_plan:
        return "Captain AI could not build a qualified bowling plan for the next overs from the current selections."

    plan_lines = []
    for item in over_plan:
        if item.recommended_bowler:
            suffix = " (debut exception)" if item.debut_exception_used else ""
            plan_lines.append(f"Over {item.over_number}: {item.recommended_bowler}{suffix}")
        else:
            plan_lines.append(f"Over {item.over_number}: unassigned")

    joined_plan = "; ".join(plan_lines)
    return (
        f"Next {len(over_plan)}-over bowling plan for {payload.bowling_team} in this {_innings_context_label(payload)} situation: "
        f"{joined_plan}. The sequence only uses legal remaining overs, blocks consecutive overs by the same bowler, and leaves overs unassigned when no qualified legal option remains."
    )


def _plan_reason(row: pd.Series, phase: str, innings_context: str, remaining_after_assignment: int) -> str:
    matchup_detail = ""
    if int(row["matchup_balls"]) > 0:
        matchup_detail = (
            f" Matchup sample: {int(row['matchup_runs'])} runs off {int(row['matchup_balls'])} balls, "
            f"{int(row['matchup_dismissals'])} dismissals."
        )
    return (
        f"Best fit for the {_phase_label(phase)} in a {innings_context} situation with {int(row['phase_innings'])} phase innings, "
        f"{int(row['wickets'])} wickets, {float(row['economy']):.2f} economy, and a wicket every {float(row['strike_rate_bowling']):.1f} balls. "
        f"Remaining overs after this assignment: {remaining_after_assignment}.{matchup_detail}"
    )


def _build_over_plan(repository: AnalyticsRepository, payload: CaptainAdviceRequest) -> tuple[list[CaptainPlanOver], dict[str, pd.DataFrame]]:
    over_numbers = _next_over_numbers(payload)
    phase_cache: dict[str, pd.DataFrame] = {}
    usage_count: dict[str, int] = {}
    remaining_overs = _normalize_remaining_overs(payload)
    over_plan: list[CaptainPlanOver] = []
    previous_bowler: str | None = None
    innings_context = _innings_context_label(payload)

    for over_number in over_numbers:
        phase = _phase_for_over_number(over_number)
        if phase not in phase_cache:
            phase_cache[phase] = _build_candidate_frame(repository, payload, phase)
        frame = phase_cache[phase]
        eligible_frame = frame[frame["eligible"]].copy()
        eligible_frame = eligible_frame[
            eligible_frame["bowler"].map(lambda bowler: remaining_overs.get(str(bowler), 0) > 0)
        ].copy()
        if previous_bowler:
            eligible_frame = eligible_frame[eligible_frame["bowler"] != previous_bowler].copy()
        if eligible_frame.empty:
            legal_pool_count = sum(1 for overs_left in remaining_overs.values() if overs_left > 0)
            reason = (
                f"No qualified legal bowler is available for this {_phase_label(phase)} over from the selected list."
            )
            if previous_bowler and remaining_overs.get(previous_bowler, 0) > 0 and legal_pool_count == 1:
                reason = (
                    f"The only bowler with legal overs left is {previous_bowler}, but consecutive overs are blocked."
                )
            elif legal_pool_count == 0:
                reason = "No selected bowler has any legal overs remaining."
            over_plan.append(
                CaptainPlanOver(
                    over_number=over_number,
                    phase=_phase_label(phase),
                    recommended_bowler=None,
                    confidence=0.28,
                    debut_exception_used=False,
                    reason=reason,
                )
            )
            previous_bowler = None
            continue

        eligible_frame["usage_penalty"] = eligible_frame["bowler"].map(lambda bowler: usage_count.get(str(bowler), 0) * 1.45)
        eligible_frame["adjusted_score"] = eligible_frame["score"] - eligible_frame["usage_penalty"]
        best = eligible_frame.sort_values(["adjusted_score", "phase_innings"], ascending=[False, False]).iloc[0]
        bowler = str(best["bowler"])
        debut_exception_used = bool(best["eligible_debut"] and not best["eligible_standard"])
        usage_count[bowler] = usage_count.get(bowler, 0) + 1
        remaining_overs[bowler] = max(remaining_overs.get(bowler, 0) - 1, 0)
        previous_bowler = bowler
        base_confidence = 0.62 if debut_exception_used else 0.74
        confidence = min(base_confidence + min(int(best["phase_innings"]), 20) * 0.007 - (usage_count[bowler] - 1) * 0.03, 0.92)
        over_plan.append(
            CaptainPlanOver(
                over_number=over_number,
                phase=_phase_label(phase),
                recommended_bowler=bowler,
                confidence=round(max(confidence, 0.35), 2),
                debut_exception_used=debut_exception_used,
                reason=_plan_reason(best, phase, innings_context, remaining_overs[bowler]),
            )
        )

    return over_plan, phase_cache


def get_captain_advice(repository: AnalyticsRepository, payload: CaptainAdviceRequest) -> CaptainResponse:
    phase = _current_phase(payload)
    over_plan, phase_cache = _build_over_plan(repository, payload)
    recommended_plan = [item for item in over_plan if item.recommended_bowler]
    candidate_frame = phase_cache.get(phase, _build_candidate_frame(repository, payload, phase))
    eligible_frame = candidate_frame[candidate_frame["eligible"]].copy()

    if not recommended_plan or eligible_frame.empty:
        explanation = (
            f"No phase-qualified bowling plan is available for the next overs from the currently selected list. "
            f"Captain AI is blocking part-time options unless they clear {PHASE_MIN_INNINGS} phase innings, or the debut exception of "
            f"{DEBUT_PHASE_MIN_INNINGS}+ phase innings with fewer than {DEBUT_TOTAL_MATCHES_LIMIT} total IPL matches."
        )
        evidence = [
            EvidenceItem(
                title="Phase eligibility guardrail",
                stat=f"{_phase_label(phase)} only",
                detail="Switch to all-player mode or choose a bowler with a stronger phase sample and legal overs remaining for a reliable recommendation.",
            )
        ]
        prompt = "\n".join(
            [
                "You are a tactical T20 cricket assistant.",
                "Return an insufficient-evidence message when phase-qualified bowlers are unavailable.",
                f"Current phase: {_phase_label(phase)}.",
                explanation,
            ]
        )
        contexts = _retrieve_context(repository, prompt, limit=3)
        return CaptainResponse(
            answer=explanation,
            over_plan=over_plan,
            evidence=evidence,
            context_ids=[item["id"] for item in contexts],
            confidence=0.32,
            fallback_used=True,
        )

    best = eligible_frame.iloc[0]
    anchor_recommendation = str(best["bowler"])
    debut_exception_used = bool(best["eligible_debut"] and not best["eligible_standard"])

    evidence = [
        EvidenceItem(
            title="Innings context",
            stat=_innings_context_label(payload),
            detail=(
                "This recommendation is tuned for first-innings control and wicket creation."
                if payload.innings == 1
                else "This recommendation is tuned for defending the chase with phase-specific control."
            ),
        ),
        EvidenceItem(
            title=f"{anchor_recommendation} {_phase_label(phase)} profile",
            stat=f"{int(best['phase_innings'])} phase innings",
            detail=(
                f"{anchor_recommendation} has {int(best['wickets'])} wickets and a {float(best['economy']):.2f} economy "
                f"for {payload.bowling_team} in the {_phase_label(phase)}."
            ),
        )
    ]

    if int(best["matchup_balls"]) > 0:
        evidence.append(
            EvidenceItem(
                title=f"{anchor_recommendation} vs current batters",
                stat=f"{int(best['matchup_dismissals'])} dismissals",
                detail=(
                    f"Against {', '.join(payload.batters[:2])}, {anchor_recommendation} has conceded "
                    f"{int(best['matchup_runs'])} off {int(best['matchup_balls'])} balls."
                ),
            )
        )

    if debut_exception_used:
        evidence.append(
            EvidenceItem(
                title="Debut exception",
                stat=f"{int(best['total_matches'])} total IPL matches",
                detail=(
                    f"{anchor_recommendation} is below the standard {PHASE_MIN_INNINGS}-innings phase threshold, "
                    f"but qualifies through the debut exception with {int(best['phase_innings'])} phase innings."
                ),
            )
        )

    evidence.append(
        EvidenceItem(
            title="Next 5 overs plan",
            stat=f"{len(recommended_plan)} qualified overs mapped",
            detail="Captain AI now sequences the upcoming overs, enforces legal remaining overs, and blocks consecutive overs by the same bowler.",
        )
    )

    prompt = "\n".join(
        [
            "You are a tactical T20 cricket assistant.",
            "Use only the evidence below and summarize the next five-over bowling plan.",
            f"Current state: innings {payload.innings}, score {payload.current_score}/{payload.wickets_lost}, over {payload.overs_completed}.{payload.balls_into_over}, venue {payload.venue}.",
            f"Innings context: {_innings_context_label(payload)}.",
            f"Current phase: {_phase_label(phase)}.",
            "Planned overs:",
            *[
                f"- Over {item.over_number}: {item.recommended_bowler or 'No qualified option'} in the {item.phase}. {item.reason}"
                for item in over_plan
            ],
            "Evidence:",
            *[f"- {item.title}: {item.stat}. {item.detail}" for item in evidence],
            "Return 2 concise sentences and mention the over-by-over plan.",
        ]
    )
    generated = _call_ollama(prompt)
    manual_answer = _build_manual_answer(over_plan, payload)
    fallback = generated is None or not any(
        item.recommended_bowler and item.recommended_bowler in generated for item in recommended_plan[:2]
    )
    answer = manual_answer if fallback else generated
    contexts = _retrieve_context(repository, prompt, limit=5)
    average_plan_confidence = sum(item.confidence for item in recommended_plan) / max(len(recommended_plan), 1)
    confidence = min(max(average_plan_confidence, 0.58) + 0.04, 0.91)
    return CaptainResponse(
        answer=answer,
        over_plan=over_plan,
        evidence=evidence[:4],
        context_ids=[item["id"] for item in contexts],
        confidence=round(confidence, 2),
        fallback_used=fallback,
    )


def chat_with_captain(repository: AnalyticsRepository, payload: CaptainAdviceRequest) -> CaptainResponse:
    return get_captain_advice(repository, payload)


def optimize_impact(repository: AnalyticsRepository, payload: ImpactOptimizationRequest) -> ImpactOptimizationResponse:
    baseline = predict(
        repository,
        payload.model_copy(
            update={
                "target_score": payload.target_score,
            }
        ),
    )
    base_probability = baseline.win_probability
    adjustments = ROLE_ADJUSTMENTS[payload.player_role]
    scenarios = []
    for band, label in [("early", "Over 4-6"), ("middle", "Over 10-13"), ("late", "Over 16-18")]:
        gain = adjustments[band]
        state_multiplier = 1.0
        if payload.innings == 2 and payload.target_score:
            current_required = payload.target_score - payload.current_score
            if current_required > 40 and label == "Over 16-18":
                state_multiplier = 1.2
        estimated = min(base_probability + gain * state_multiplier, 99.0)
        scenarios.append(
            ImpactScenario(
                over_band=label,
                estimated_win_probability=round(estimated, 2),
                marginal_gain=round(estimated - base_probability, 2),
                reason=f"{payload.player_role.replace('_', ' ')} profiles historically create their biggest marginal value in this band.",
            )
        )
    best = max(scenarios, key=lambda item: item.marginal_gain)
    return ImpactOptimizationResponse(
        recommended_over_band=best.over_band,
        best_gain=best.marginal_gain,
        assumption_note="This optimizer is a simulation-driven heuristic on top of the win model because the raw dataset does not encode official impact-substitution events.",
        scenarios=scenarios,
    )
