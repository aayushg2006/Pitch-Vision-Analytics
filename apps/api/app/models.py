from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TossImpactDatum(BaseModel):
    outcome: str
    win_rate: float
    matches: int


class PhaseComparisonDatum(BaseModel):
    phase: str
    winners: float
    losers: float


class LeaderboardEntry(BaseModel):
    rank: int
    name: str
    value: int


class TeamTossImpactDatum(BaseModel):
    team: str
    toss_won_matches: int
    toss_lost_matches: int
    win_pct_after_winning_toss: float
    win_pct_after_losing_toss: float


class TeamInningsImpactDatum(BaseModel):
    team: str
    batting_first_matches: int
    bowling_first_matches: int
    win_pct_batting_first: float
    win_pct_bowling_first: float


class TeamMilestoneImpactDatum(BaseModel):
    team: str
    chasing_200_plus_matches: int
    defending_150_or_lower_matches: int
    win_pct_chasing_200_plus: float
    win_pct_defending_150_or_lower: float


class DashboardSummaryResponse(BaseModel):
    generated_at: str
    scope: str
    toss_impact: list[TossImpactDatum]
    team_toss_impact: list[TeamTossImpactDatum]
    team_innings_impact: list[TeamInningsImpactDatum]
    team_milestone_impact: list[TeamMilestoneImpactDatum]
    phase_comparison: list[PhaseComparisonDatum]
    top_batters: list[LeaderboardEntry]
    top_bowlers: list[LeaderboardEntry]
    surprising_finding: str


class DataProofResponse(BaseModel):
    generated_at: str
    source_file: str
    cleaning_script: str
    rows_in_raw: int
    matches_in_dataset: int
    seasons: list[str]
    mappings: dict
    derived_fields: list[str]


class OptionItem(BaseModel):
    label: str
    value: str


class TeamPlayerPool(BaseModel):
    team: str
    players: list[OptionItem]


class OptionsResponse(BaseModel):
    batters: list[OptionItem]
    bowlers: list[OptionItem]
    teams: list[OptionItem]
    venues: list[OptionItem]
    captain_batters_by_team: list[TeamPlayerPool]
    captain_bowlers_by_team: list[TeamPlayerPool]


class WinPredictRequest(BaseModel):
    innings: Literal[1, 2]
    current_score: int = Field(ge=0)
    wickets_lost: int = Field(ge=0, le=10)
    overs_completed: int = Field(ge=0, le=20)
    balls_into_over: int = Field(ge=0, le=5)
    target_score: int | None = Field(default=None, ge=1)
    batting_team: str
    bowling_team: str
    venue: str


class WinPredictResponse(BaseModel):
    win_probability: float
    confidence_band: tuple[float, float]
    likely_winner: str
    top_factors: list[str]
    model_version: str
    evaluation_stamp: str


class HeadToHeadRequest(BaseModel):
    batter: str
    bowler: str


class HeadToHeadResponse(BaseModel):
    batter: str
    bowler: str
    runs: int
    balls: int
    dismissals: int
    strike_rate: float
    boundaries: int
    sixes: int


class PartnershipRequest(BaseModel):
    batter_one: str
    batter_two: str


class PartnershipResponse(BaseModel):
    pair: tuple[str, str]
    runs: int
    balls: int
    run_rate: float
    average_partnership: float
    shared_innings: int
    synergy_index: float


class PressureFilter(BaseModel):
    role: Literal["batter", "bowler"] = "batter"
    player_name: str | None = None
    limit: int = Field(default=10, ge=1, le=30)


class PressurePlayerRow(BaseModel):
    player_name: str
    innings: int
    matches_played: int
    metric_one: float
    metric_two: float
    pressure_index: float


class PressureOverviewResponse(BaseModel):
    role: Literal["batter", "bowler"]
    rows: list[PressurePlayerRow]


class EvidenceItem(BaseModel):
    title: str
    stat: str
    detail: str


class CaptainPlanOver(BaseModel):
    over_number: int
    phase: str
    recommended_bowler: str | None = None
    confidence: float
    debut_exception_used: bool = False
    reason: str


class CaptainAdviceRequest(BaseModel):
    innings: Literal[1, 2]
    current_score: int = Field(ge=0)
    wickets_lost: int = Field(ge=0, le=10)
    overs_completed: int = Field(ge=0, le=20)
    balls_into_over: int = Field(ge=0, le=5)
    target_score: int | None = Field(default=None, ge=1)
    batting_team: str
    bowling_team: str
    venue: str
    batters: list[str] = Field(default_factory=list)
    available_bowlers: list[str] = Field(default_factory=list)
    remaining_overs_by_bowler: dict[str, int] = Field(default_factory=dict)


class CaptainResponse(BaseModel):
    answer: str
    over_plan: list[CaptainPlanOver]
    evidence: list[EvidenceItem]
    context_ids: list[str]
    confidence: float
    fallback_used: bool


class ImpactOptimizationRequest(BaseModel):
    innings: Literal[1, 2]
    current_score: int = Field(ge=0)
    wickets_lost: int = Field(ge=0, le=10)
    overs_completed: int = Field(ge=0, le=20)
    balls_into_over: int = Field(ge=0, le=5)
    target_score: int | None = Field(default=None, ge=1)
    batting_team: str
    bowling_team: str
    venue: str
    player_role: Literal["finisher", "anchor", "spinner", "pacer", "all_rounder"]


class ImpactScenario(BaseModel):
    over_band: str
    estimated_win_probability: float
    marginal_gain: float
    reason: str


class ImpactOptimizationResponse(BaseModel):
    recommended_over_band: str
    best_gain: float
    assumption_note: str
    scenarios: list[ImpactScenario]
