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


class ProjectedScore(BaseModel):
    linear_regression: float
    svr: float
    rule_of_thumb: str


class ModelOpinion(BaseModel):
    model: str
    probability: float


class WinPredictResponse(BaseModel):
    win_probability: float
    confidence_band: tuple[float, float]
    likely_winner: str
    top_factors: list[str]
    model_version: str
    evaluation_stamp: str
    projected_score: ProjectedScore | None = None
    model_opinions: list[ModelOpinion] | None = None
    decision_path: list[str] | None = None


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
    bowler_type: str | None = None
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


class ChartInfo(BaseModel):
    name: str
    title: str
    description: str


class RegressionMetrics(BaseModel):
    r2: float
    mae: float
    rmse: float


class SimpleRegressionReport(RegressionMetrics):
    intercept: float
    slope: float


class MultipleRegressionReport(RegressionMetrics):
    intercept: float
    coefficients: dict[str, float]


class ScoreModelsReport(BaseModel):
    holdout_seasons: list[int]
    simple: SimpleRegressionReport
    multiple: MultipleRegressionReport
    svr: RegressionMetrics


class ClassificationMetrics(BaseModel):
    accuracy: float
    roc_auc: float


class LeaderboardRow(ClassificationMetrics):
    model: str


class TuningRow(BaseModel):
    model: str
    method: str
    best_params: dict[str, float | int | str]
    before: ClassificationMetrics
    after: ClassificationMetrics


class ChaseModelsReport(BaseModel):
    holdout_seasons: list[int]
    features: list[str]
    leaderboard: list[LeaderboardRow]
    tuning: list[TuningRow]
    tree_chart: str | None = None
    tree_rules: str


class FreakInnings(BaseModel):
    batting_team: str
    fielding_team: str
    venue: str
    season: int
    innings: int
    score: str
    run_rate: float
    fours: int
    sixes: int
    anomaly_score: float


class ElbowPoint(BaseModel):
    k: int
    inertia: float


class BatterCluster(BaseModel):
    cluster: int
    style: str
    size: int
    centroid: dict[str, float]
    top_players: list[str]


class BatterPoint(BaseModel):
    player: str
    style: str
    x: float
    y: float
    strike_rate: float
    average: float


class BatterStylesReport(BaseModel):
    k: int
    features: list[str]
    elbow: list[ElbowPoint]
    explained_variance: list[float]
    clusters: list[BatterCluster]
    points: list[BatterPoint]


class CoachPolicyRow(BaseModel):
    balls_left: str
    required_rate: str
    wickets_in_hand: str
    best_action: str
    q_values: dict[str, float]
    visits: int


class LearningPoint(BaseModel):
    episode: int
    win_rate: float


class ChaseCoachReport(BaseModel):
    episodes: int
    actions: list[str]
    balls_buckets: list[str]
    rate_buckets: list[str]
    wickets_buckets: list[str]
    balls_edges: list[int]
    rate_edges: list[float]
    wickets_edges: list[int]
    outcome_probabilities: dict[str, dict[str, dict[str, float]]]
    policy: list[CoachPolicyRow]
    test_chases: int
    learning_curve: list[LearningPoint]


class InsightsResponse(BaseModel):
    generated_at: str
    charts: list[ChartInfo]
    score_models: ScoreModelsReport
    chase_models: ChaseModelsReport
    freak_innings: list[FreakInnings]
    batter_styles: BatterStylesReport
    chase_coach: ChaseCoachReport
