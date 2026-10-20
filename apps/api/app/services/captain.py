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
PREFERRED_OLLAMA_MODELS = ("tinyllama", "llama3.2:1b", "qwen2.5:3b")
SHRINK_BALLS = 60
MATCHUP_SHRINK_BALLS = 30
WICKET_RUN_VALUE = 7.0
PACE_STYLE_WORDS = ("fast", "medium")
SPIN_STYLE_WORDS = ("break", "orthodox", "spin", "chinaman")
# Run-equivalent bonuses for the death overs: spinners bowl out early, pacers close the innings.
SPIN_FIT_BY_OVER = {16: 0.3, 17: -0.3, 18: -0.9, 19: -1.5, 20: -2.0}
PACE_FIT_BY_OVER = {16: 0.0, 17: 0.4, 18: 0.8, 19: 1.2, 20: 1.5}


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


_resolved_ollama_model: str | None = None


def _resolve_ollama_model() -> str | None:
    """Pick a model that is already installed locally; never pulls a new one."""
    global _resolved_ollama_model
    if _resolved_ollama_model is not None:
        return _resolved_ollama_model
    settings = get_settings()
    try:
        response = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        installed = [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return None
    if not installed:
        return None
    _resolved_ollama_model = next(
        (
            name
            for candidate in (settings.ollama_model, *PREFERRED_OLLAMA_MODELS)
            for name in installed
            if name in (candidate, f"{candidate}:latest")
        ),
        installed[0],
    )
    return _resolved_ollama_model


def _call_ollama(prompt: str) -> str | None:
    settings = get_settings()
    model = _resolve_ollama_model()
    if model is None:
        return None
    try:
        response = requests.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=90,
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


def _bowler_types(repository: AnalyticsRepository, bowlers: list[str]) -> dict[str, tuple[str, str]]:
    """Map each bowler to (label, category) where category is "pace" or "spin".

    Known bowling styles come from player metadata. Everyone else is classified by when they usually bowl:
    bowlers who live in overs 7-15 are treated like spinners, new-ball and death bowlers like pacers.
    """
    if not bowlers:
        return {}
    styles = repository.fetch_dataframe(
        f"SELECT player_name, bowling_style FROM player_metadata WHERE player_name IN ({_placeholder_list(bowlers)})",
        bowlers,
    )
    known = {
        str(row.player_name): str(row.bowling_style).lower()
        for row in styles.itertuples(index=False)
        if str(row.bowling_style).lower() not in ("", "none", "unknown", "nan")
    }
    usage = repository.fetch_dataframe(
        f"""
        SELECT
            bowler,
            SUM(legal_delivery) AS balls,
            SUM(CASE WHEN phase = 'middle' THEN legal_delivery ELSE 0 END) AS middle_balls,
            SUM(CASE WHEN phase = 'death' THEN legal_delivery ELSE 0 END) AS death_balls
        FROM deliveries
        WHERE bowler IN ({_placeholder_list(bowlers)})
        GROUP BY bowler
        """,
        bowlers,
    ).set_index("bowler")

    types: dict[str, tuple[str, str]] = {}
    for bowler in bowlers:
        style = known.get(bowler, "")
        if any(word in style for word in SPIN_STYLE_WORDS):
            types[bowler] = ("spinner", "spin")
        elif any(word in style for word in PACE_STYLE_WORDS):
            types[bowler] = ("pacer", "pace")
        elif bowler in usage.index and usage.loc[bowler, "balls"] > 0:
            balls = float(usage.loc[bowler, "balls"])
            middle_share = float(usage.loc[bowler, "middle_balls"]) / balls
            death_share = float(usage.loc[bowler, "death_balls"]) / balls
            if middle_share >= 0.55 and death_share < 0.3:
                types[bowler] = ("middle-overs bowler", "spin")
            else:
                types[bowler] = ("new-ball / death bowler", "pace")
        else:
            types[bowler] = ("unknown type", "")
    return types


def _type_fit(category: str, over_number: int) -> float:
    """Run-equivalent bonus: spin early and through the middle, pace with the new ball and saved for the last overs."""
    if category == "spin":
        return SPIN_FIT_BY_OVER.get(over_number, 0.8 if over_number > 6 else -0.8)
    if category == "pace":
        return PACE_FIT_BY_OVER.get(over_number, -0.6 if over_number > 6 else 0.8)
    return 0.0


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

    # Situation record: the same phase in the same innings (bowling first vs defending a target), for any team.
    situation_rows = repository.fetch_dataframe(
        f"""
        SELECT
            bowler,
            SUM(CASE WHEN innings = ? THEN runs_total ELSE 0 END) AS situation_runs,
            SUM(CASE WHEN innings = ? THEN legal_delivery ELSE 0 END) AS situation_balls,
            SUM(CASE WHEN innings = ? THEN bowler_wicket ELSE 0 END) AS situation_wickets,
            SUM(runs_total) AS career_phase_runs,
            SUM(legal_delivery) AS career_phase_balls,
            SUM(bowler_wicket) AS career_phase_wickets
        FROM deliveries
        WHERE phase = ?
          AND bowler IN ({_placeholder_list(payload.available_bowlers)})
        GROUP BY bowler
        """,
        [payload.innings, payload.innings, payload.innings, phase, *payload.available_bowlers],
    )
    league = repository.fetch_dataframe(
        "SELECT SUM(runs_total) / SUM(legal_delivery) AS runs_per_ball, SUM(bowler_wicket) / SUM(legal_delivery) AS wickets_per_ball "
        "FROM deliveries WHERE phase = ?",
        [phase],
    ).iloc[0]

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

    candidate_frame = (
        phase_rows.merge(situation_rows, on="bowler", how="left")
        .merge(total_rows, on="bowler", how="left")
        .merge(matchups, on="bowler", how="left")
    )
    count_columns = [
        "matchup_runs",
        "matchup_balls",
        "matchup_dismissals",
        "situation_runs",
        "situation_balls",
        "situation_wickets",
        "career_phase_runs",
        "career_phase_balls",
        "career_phase_wickets",
    ]
    candidate_frame[count_columns] = candidate_frame[count_columns].fillna(0)
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

    # Shrink small samples: situation record -> career phase record -> league phase average.
    prior_runs = (candidate_frame["career_phase_runs"] + SHRINK_BALLS * float(league["runs_per_ball"])) / (
        candidate_frame["career_phase_balls"] + SHRINK_BALLS
    )
    prior_wickets = (candidate_frame["career_phase_wickets"] + SHRINK_BALLS * float(league["wickets_per_ball"])) / (
        candidate_frame["career_phase_balls"] + SHRINK_BALLS
    )
    runs_per_ball = (candidate_frame["situation_runs"] + SHRINK_BALLS * prior_runs) / (candidate_frame["situation_balls"] + SHRINK_BALLS)
    wickets_per_ball = (candidate_frame["situation_wickets"] + SHRINK_BALLS * prior_wickets) / (
        candidate_frame["situation_balls"] + SHRINK_BALLS
    )
    # Blend in the head-to-head record against the batters at the crease.
    matchup_weight = candidate_frame["matchup_balls"] / (candidate_frame["matchup_balls"] + MATCHUP_SHRINK_BALLS)
    runs_per_ball = (1 - matchup_weight) * runs_per_ball + matchup_weight * (
        candidate_frame["matchup_runs"] / candidate_frame["matchup_balls"].clip(lower=1)
    )
    wickets_per_ball = (1 - matchup_weight) * wickets_per_ball + matchup_weight * (
        candidate_frame["matchup_dismissals"] / candidate_frame["matchup_balls"].clip(lower=1)
    )
    candidate_frame["situation_economy"] = runs_per_ball * 6
    candidate_frame["situation_wickets_per_over"] = wickets_per_ball * 6
    candidate_frame["score"] = candidate_frame["situation_wickets_per_over"] * WICKET_RUN_VALUE - candidate_frame["situation_economy"]

    types = _bowler_types(repository, candidate_frame["bowler"].astype(str).tolist())
    candidate_frame["bowler_type"] = candidate_frame["bowler"].map(lambda bowler: types.get(str(bowler), ("unknown type", ""))[0])
    candidate_frame["type_category"] = candidate_frame["bowler"].map(lambda bowler: types.get(str(bowler), ("unknown type", ""))[1])
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


def _type_note(category: str, over_number: int) -> str:
    if category == "spin" and over_number in (16, 17):
        return " Spin is used early in the death so pace is saved for the final overs."
    if category == "spin" and over_number >= 18:
        return " A spinner this late only because no stronger legal pace option fits this over."
    if category == "pace" and over_number >= 18:
        return " Pace is saved for the last overs."
    if category == "spin" and 7 <= over_number <= 15:
        return " Middle overs suit this bowling type."
    if category == "pace" and over_number <= 6:
        return " New-ball overs suit this bowling type."
    return ""


def _plan_reason(row: pd.Series, phase: str, innings_context: str, remaining_after_assignment: int, over_number: int) -> str:
    matchup_detail = ""
    if int(row["matchup_balls"]) > 0:
        matchup_detail = (
            f" Matchup sample: {int(row['matchup_runs'])} runs off {int(row['matchup_balls'])} balls, "
            f"{int(row['matchup_dismissals'])} dismissals."
        )
    situation_balls = int(row["situation_balls"])
    evidence = f"from {situation_balls} balls in this situation" if situation_balls else "estimated from their overall phase record"
    wicket_every = 6 / max(float(row["situation_wickets_per_over"]), 0.01)
    return (
        f"{str(row['bowler_type']).capitalize()}. In the {_phase_label(phase)} of a {innings_context}, expected to concede "
        f"{float(row['situation_economy']):.2f} an over with a wicket every {wicket_every:.1f} balls ({evidence})."
        f"{_type_note(str(row['type_category']), over_number)}"
        f" Remaining overs after this assignment: {remaining_after_assignment}.{matchup_detail}"
    )


def _unassigned_reason(phase: str, previous_bowler: str | None, remaining_overs: dict[str, int]) -> str:
    legal_pool_count = sum(1 for overs_left in remaining_overs.values() if overs_left > 0)
    if legal_pool_count == 0:
        return "No selected bowler has any legal overs remaining."
    if previous_bowler and remaining_overs.get(previous_bowler, 0) > 0 and legal_pool_count == 1:
        return f"The only bowler with legal overs left is {previous_bowler}, but consecutive overs are blocked."
    return f"No qualified legal bowler is available for this {_phase_label(phase)} over from the selected list."


def _search_over_plan(
    over_numbers: list[int],
    candidates: list[dict[str, float]],
    remaining_overs: dict[str, int],
) -> list[str | None]:
    """Try every legal sequence: cover as many overs as possible first, then maximise the total over value."""
    bowlers = sorted(remaining_overs)
    index_of = {bowler: index for index, bowler in enumerate(bowlers)}
    memo: dict[tuple[int, str | None, tuple[int, ...]], tuple[int, float, tuple[str | None, ...]]] = {}

    def solve(position: int, previous: str | None, remaining: tuple[int, ...]) -> tuple[int, float, tuple[str | None, ...]]:
        if position == len(over_numbers):
            return 0, 0.0, ()
        key = (position, previous, remaining)
        if key in memo:
            return memo[key]
        covered, value, rest = solve(position + 1, None, remaining)
        best = (covered, value, (None, *rest))
        for bowler, over_value in candidates[position].items():
            slot = index_of[bowler]
            if bowler == previous or remaining[slot] == 0:
                continue
            next_remaining = remaining[:slot] + (remaining[slot] - 1,) + remaining[slot + 1 :]
            covered, value, rest = solve(position + 1, bowler, next_remaining)
            option = (covered + 1, value + over_value, (bowler, *rest))
            if (option[0], option[1]) > (best[0], best[1]):
                best = option
        memo[key] = best
        return best

    return list(solve(0, None, tuple(remaining_overs[bowler] for bowler in bowlers))[2])


def _build_over_plan(repository: AnalyticsRepository, payload: CaptainAdviceRequest) -> tuple[list[CaptainPlanOver], dict[str, pd.DataFrame]]:
    over_numbers = _next_over_numbers(payload)
    phase_cache: dict[str, pd.DataFrame] = {}
    remaining_overs = _normalize_remaining_overs(payload)
    innings_context = _innings_context_label(payload)

    candidates: list[dict[str, float]] = []
    for over_number in over_numbers:
        phase = _phase_for_over_number(over_number)
        if phase not in phase_cache:
            phase_cache[phase] = _build_candidate_frame(repository, payload, phase)
        frame = phase_cache[phase]
        eligible_frame = frame[frame["eligible"]] if not frame.empty else frame
        candidates.append(
            {
                str(row.bowler): float(row.score) + _type_fit(str(row.type_category), over_number)
                for row in eligible_frame.itertuples(index=False)
                if remaining_overs.get(str(row.bowler), 0) > 0
            }
        )

    assignments = _search_over_plan(over_numbers, candidates, remaining_overs)
    over_plan: list[CaptainPlanOver] = []
    previous_bowler: str | None = None
    for over_number, bowler in zip(over_numbers, assignments):
        phase = _phase_for_over_number(over_number)
        if bowler is None:
            over_plan.append(
                CaptainPlanOver(
                    over_number=over_number,
                    phase=_phase_label(phase),
                    recommended_bowler=None,
                    confidence=0.28,
                    debut_exception_used=False,
                    reason=_unassigned_reason(phase, previous_bowler, remaining_overs),
                )
            )
            previous_bowler = None
            continue

        frame = phase_cache[phase]
        best = frame[frame["bowler"] == bowler].iloc[0]
        debut_exception_used = bool(best["eligible_debut"] and not best["eligible_standard"])
        remaining_overs[bowler] = max(remaining_overs.get(bowler, 0) - 1, 0)
        previous_bowler = bowler
        base_confidence = 0.62 if debut_exception_used else 0.7
        confidence = min(base_confidence + min(float(best["situation_balls"]), 240) / 240 * 0.2, 0.92)
        over_plan.append(
            CaptainPlanOver(
                over_number=over_number,
                phase=_phase_label(phase),
                recommended_bowler=bowler,
                bowler_type=str(best["bowler_type"]),
                confidence=round(confidence, 2),
                debut_exception_used=debut_exception_used,
                reason=_plan_reason(best, phase, innings_context, remaining_overs[bowler], over_number),
            )
        )

    return over_plan, phase_cache


def get_captain_advice(repository: AnalyticsRepository, payload: CaptainAdviceRequest) -> CaptainResponse:
    phase = _current_phase(payload)
    over_plan, phase_cache = _build_over_plan(repository, payload)
    recommended_plan = [item for item in over_plan if item.recommended_bowler]
    if not recommended_plan:
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

    # Evidence is about the bowler of the first planned over, using that over's phase.
    first_over = recommended_plan[0]
    phase = _phase_for_over_number(first_over.over_number)
    candidate_frame = phase_cache[phase]
    best = candidate_frame[candidate_frame["bowler"] == first_over.recommended_bowler].iloc[0]
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
            stat=f"{int(best['phase_innings'])} phase innings · {best['bowler_type']}",
            detail=(
                f"{anchor_recommendation} has {int(best['wickets'])} wickets and a {float(best['economy']):.2f} economy "
                f"for {payload.bowling_team} in the {_phase_label(phase)}. In this exact situation across their career they "
                f"are expected to concede {float(best['situation_economy']):.2f} an over."
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
            detail="Captain AI searches every legal order of the upcoming overs so no quota is stranded, bowls spin before the final overs, saves pace for the end and never gives one bowler consecutive overs.",
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
