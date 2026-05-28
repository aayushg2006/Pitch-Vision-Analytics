export type TossImpactDatum = {
  outcome: string;
  win_rate: number;
  matches: number;
};

export type PhaseComparisonDatum = {
  phase: string;
  winners: number;
  losers: number;
};

export type LeaderboardEntry = {
  rank: number;
  name: string;
  value: number;
};

export type TeamTossImpactDatum = {
  team: string;
  toss_won_matches: number;
  toss_lost_matches: number;
  win_pct_after_winning_toss: number;
  win_pct_after_losing_toss: number;
};

export type TeamInningsImpactDatum = {
  team: string;
  batting_first_matches: number;
  bowling_first_matches: number;
  win_pct_batting_first: number;
  win_pct_bowling_first: number;
};

export type TeamMilestoneImpactDatum = {
  team: string;
  chasing_200_plus_matches: number;
  defending_150_or_lower_matches: number;
  win_pct_chasing_200_plus: number;
  win_pct_defending_150_or_lower: number;
};

export type DashboardSummaryResponse = {
  generated_at: string;
  scope: string;
  toss_impact: TossImpactDatum[];
  team_toss_impact: TeamTossImpactDatum[];
  team_innings_impact: TeamInningsImpactDatum[];
  team_milestone_impact: TeamMilestoneImpactDatum[];
  phase_comparison: PhaseComparisonDatum[];
  top_batters: LeaderboardEntry[];
  top_bowlers: LeaderboardEntry[];
  surprising_finding: string;
};

export type DataProofResponse = {
  generated_at: string;
  source_file: string;
  cleaning_script: string;
  rows_in_raw: number;
  matches_in_dataset: number;
  seasons: string[];
  mappings: Record<string, Record<string, string>>;
  derived_fields: string[];
};

export type OptionItem = {
  label: string;
  value: string;
};

export type TeamPlayerPool = {
  team: string;
  players: OptionItem[];
};

export type OptionsResponse = {
  batters: OptionItem[];
  bowlers: OptionItem[];
  teams: OptionItem[];
  venues: OptionItem[];
  captain_batters_by_team: TeamPlayerPool[];
  captain_bowlers_by_team: TeamPlayerPool[];
};

export type WinPredictRequest = {
  innings: 1 | 2;
  current_score: number;
  wickets_lost: number;
  overs_completed: number;
  balls_into_over: number;
  target_score?: number;
  batting_team: string;
  bowling_team: string;
  venue: string;
};

export type WinPredictResponse = {
  win_probability: number;
  confidence_band: [number, number];
  likely_winner: string;
  top_factors: string[];
  model_version: string;
  evaluation_stamp: string;
};

export type HeadToHeadRequest = {
  batter: string;
  bowler: string;
};

export type HeadToHeadResponse = {
  batter: string;
  bowler: string;
  runs: number;
  balls: number;
  dismissals: number;
  strike_rate: number;
  boundaries: number;
  sixes: number;
};

export type PartnershipRequest = {
  batter_one: string;
  batter_two: string;
};

export type PartnershipResponse = {
  pair: [string, string];
  runs: number;
  balls: number;
  run_rate: number;
  average_partnership: number;
  shared_innings: number;
  synergy_index: number;
};

export type PressureFilter = {
  role: "batter" | "bowler";
  player_name?: string;
  limit: number;
};

export type PressurePlayerRow = {
  player_name: string;
  innings: number;
  matches_played: number;
  metric_one: number;
  metric_two: number;
  pressure_index: number;
};

export type PressureOverviewResponse = {
  role: "batter" | "bowler";
  rows: PressurePlayerRow[];
};

export type CaptainAdviceRequest = {
  innings: 1 | 2;
  current_score: number;
  wickets_lost: number;
  overs_completed: number;
  balls_into_over: number;
  target_score?: number;
  batting_team: string;
  bowling_team: string;
  venue: string;
  batters: string[];
  available_bowlers: string[];
  remaining_overs_by_bowler: Record<string, number>;
};

export type EvidenceItem = {
  title: string;
  stat: string;
  detail: string;
};

export type CaptainPlanOver = {
  over_number: number;
  phase: string;
  recommended_bowler?: string | null;
  confidence: number;
  debut_exception_used: boolean;
  reason: string;
};

export type CaptainResponse = {
  answer: string;
  over_plan: CaptainPlanOver[];
  evidence: EvidenceItem[];
  context_ids: string[];
  confidence: number;
  fallback_used: boolean;
};

export type ImpactOptimizationRequest = {
  innings: 1 | 2;
  current_score: number;
  wickets_lost: number;
  overs_completed: number;
  balls_into_over: number;
  target_score?: number;
  batting_team: string;
  bowling_team: string;
  venue: string;
  player_role: "finisher" | "anchor" | "spinner" | "pacer" | "all_rounder";
};

export type ImpactScenario = {
  over_band: string;
  estimated_win_probability: number;
  marginal_gain: number;
  reason: string;
};

export type ImpactOptimizationResponse = {
  recommended_over_band: string;
  best_gain: number;
  assumption_note: string;
  scenarios: ImpactScenario[];
};
