from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ipl-matches-dataset.csv"
PLAYER_METADATA_PATH = ROOT_DIR / "data" / "raw" / "player_metadata.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

VENUE_ALIASES = {
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
}

CITY_ALIASES = {
    "Bangalore": "Bengaluru",
}

BOWLER_EXCLUDED_WICKETS = {
    "",
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}

BATTER_EXCLUDED_WICKETS = {
    "",
    "retired hurt",
}

MIN_PRI_INNINGS = 50


@dataclass
class PipelineArtifacts:
    deliveries: pd.DataFrame
    matches: pd.DataFrame
    head_to_head: pd.DataFrame
    partnerships: pd.DataFrame
    pressure_batting: pd.DataFrame
    pressure_bowling: pd.DataFrame
    player_metadata: pd.DataFrame
    dashboard_summary: dict
    audit_summary: dict


def calculate_win_percentage(wins: float, matches: float) -> float:
    return round(float(wins * 100 / matches), 2) if matches else 0.0


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def canonicalize_series(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.fillna("").replace(mapping)


def parse_season(season_label: str) -> tuple[int, str]:
    if "/" in season_label:
        first, _ = season_label.split("/", 1)
        return int(first), season_label
    return int(season_label), season_label


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    numeric_columns = [
        "match_id",
        "win_by_runs",
        "win_by_wickets",
        "innings",
        "over",
        "ball",
        "runs_batter",
        "runs_extras",
        "runs_total",
        "extras_wides",
        "extras_noballs",
        "extras_byes",
        "extras_legbyes",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    object_columns = df.select_dtypes(include="object").columns.tolist()
    for column in object_columns:
        df[column] = df[column].fillna("").astype(str)
    for column in ["team1", "team2", "toss_winner", "winner", "batting_team"]:
        df[column] = canonicalize_series(df[column], TEAM_ALIASES)
    df["venue"] = canonicalize_series(df["venue"], VENUE_ALIASES)
    df["city"] = canonicalize_series(df["city"], CITY_ALIASES)
    df["winner"] = df["winner"].replace({"tie": ""})
    seasons = df["season"].astype(str).map(parse_season)
    df["season_year"] = seasons.map(lambda item: item[0])
    df["season_label"] = seasons.map(lambda item: item[1])
    return df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)


def enrich_match_state(df: pd.DataFrame) -> pd.DataFrame:
    wicket_kind = df["wicket_kind"].fillna("").str.strip().str.lower()
    df["legal_delivery"] = ((df["extras_wides"] == 0) & (df["extras_noballs"] == 0)).astype(int)
    df["wicket_fell"] = (~wicket_kind.isin(BATTER_EXCLUDED_WICKETS)).astype(int)
    df["bowler_wicket"] = (~wicket_kind.isin(BOWLER_EXCLUDED_WICKETS)).astype(int)

    innings_group = df.groupby(["match_id", "innings"], sort=False)
    df["runs_before"] = innings_group["runs_total"].cumsum() - df["runs_total"]
    df["runs_after"] = innings_group["runs_total"].cumsum()
    df["legal_balls_after"] = innings_group["legal_delivery"].cumsum()
    df["legal_balls_before"] = df["legal_balls_after"] - df["legal_delivery"]
    df["wickets_before"] = innings_group["wicket_fell"].cumsum() - df["wicket_fell"]
    df["wickets_after"] = innings_group["wicket_fell"].cumsum()
    df["overs_completed_before"] = np.floor(df["legal_balls_before"] / 6).astype(int)
    df["balls_into_over_before"] = (df["legal_balls_before"] % 6).astype(int)
    df["current_rr_before"] = np.where(
        df["legal_balls_before"] > 0,
        df["runs_before"] * 6 / df["legal_balls_before"],
        0.0,
    )
    df["wickets_in_hand_before"] = 10 - df["wickets_before"]
    df["phase"] = np.select(
        [df["over"] <= 5, df["over"].between(6, 14), df["over"] >= 15],
        ["powerplay", "middle", "death"],
        default="middle",
    )

    innings_totals = (
        df.groupby(["match_id", "innings"], as_index=False)
        .agg(
            batting_team=("batting_team", "first"),
            innings_runs=("runs_total", "sum"),
            innings_wickets=("wicket_fell", "sum"),
        )
    )
    first_innings = innings_totals[innings_totals["innings"] == 1][["match_id", "innings_runs", "batting_team"]].rename(
        columns={"innings_runs": "first_innings_runs", "batting_team": "first_innings_team"}
    )
    df = df.merge(first_innings, on="match_id", how="left")
    df["target_score"] = np.where(df["innings"] == 2, df["first_innings_runs"] + 1, np.nan)
    df["remaining_balls_before"] = np.where(df["innings"] == 2, 120 - df["legal_balls_before"], 120 - df["legal_balls_before"])
    df["required_runs_before"] = np.where(
        df["innings"] == 2,
        np.maximum(df["target_score"] - df["runs_before"], 0),
        np.nan,
    )
    df["required_rr_before"] = np.where(
        (df["innings"] == 2) & (df["remaining_balls_before"] > 0),
        df["required_runs_before"] * 6 / df["remaining_balls_before"],
        np.nan,
    )
    df["batting_team_won"] = (df["batting_team"] == df["winner"]).astype(int)
    df["fielding_team"] = np.where(df["team1"] == df["batting_team"], df["team2"], df["team1"])

    chase_pressure = (
        ((df["required_rr_before"] >= 10.0) & (df["legal_balls_before"] >= 60))
        | ((df["wickets_in_hand_before"] <= 5) & (df["legal_balls_before"] >= 84))
        | ((df["required_runs_before"] >= 35) & (df["remaining_balls_before"] <= 24))
    )
    first_innings_pressure = (
        ((df["current_rr_before"] >= 9.0) & (df["legal_balls_before"] >= 84))
        | ((df["wickets_in_hand_before"] <= 4) & (df["legal_balls_before"] >= 72))
    )
    df["is_pressure"] = np.where(df["innings"] == 2, chase_pressure, first_innings_pressure).astype(int)
    df["pressure_state"] = np.where(df["is_pressure"] == 1, "high", "normal")
    return df


def build_match_table(df: pd.DataFrame) -> pd.DataFrame:
    match_level = (
        df.groupby("match_id", as_index=False)
        .agg(
            date=("date", "first"),
            season_year=("season_year", "first"),
            season_label=("season_label", "first"),
            event=("event", "first"),
            venue=("venue", "first"),
            city=("city", "first"),
            team1=("team1", "first"),
            team2=("team2", "first"),
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            winner=("winner", "first"),
            player_of_match=("player_of_match", "first"),
            win_by_runs=("win_by_runs", "max"),
            win_by_wickets=("win_by_wickets", "max"),
        )
    )
    innings_totals = (
        df.groupby(["match_id", "innings"], as_index=False)
        .agg(batting_team=("batting_team", "first"), innings_runs=("runs_total", "sum"))
    )
    innings_runs = (
        innings_totals.pivot(index="match_id", columns="innings", values="innings_runs")
        .reset_index()
        .rename(columns={1: "first_innings_runs", 2: "second_innings_runs"})
    )
    innings_teams = (
        innings_totals.pivot(index="match_id", columns="innings", values="batting_team")
        .reset_index()
        .rename(columns={1: "first_innings_team", 2: "second_innings_team"})
    )
    match_level = match_level.merge(innings_runs, on="match_id", how="left").merge(innings_teams, on="match_id", how="left")
    match_level["toss_winner_won"] = (
        (match_level["winner"] != "") & (match_level["toss_winner"] == match_level["winner"])
    ).astype(int)
    match_level["toss_loser_won"] = (
        (match_level["winner"] != "") & (match_level["toss_winner"] != match_level["winner"])
    ).astype(int)
    match_level["target_score"] = match_level["first_innings_runs"] + 1
    return match_level


def load_player_metadata() -> pd.DataFrame:
    seed_metadata = pd.read_csv(PLAYER_METADATA_PATH)
    seed_metadata["player_name"] = seed_metadata["player_name"].fillna("")
    return seed_metadata


def build_player_metadata(df: pd.DataFrame) -> pd.DataFrame:
    seed = load_player_metadata()
    batting = (
        df.groupby("batter", as_index=False)
        .agg(total_runs=("runs_batter", "sum"), balls_faced=("legal_delivery", "sum"))
        .rename(columns={"batter": "player_name"})
    )
    bowling = (
        df.groupby("bowler", as_index=False)
        .agg(
            credited_wickets=("bowler_wicket", "sum"),
            balls_bowled=("legal_delivery", "sum"),
            runs_conceded=("runs_total", "sum"),
        )
        .rename(columns={"bowler": "player_name"})
    )
    players = pd.DataFrame({"player_name": pd.unique(pd.concat([batting["player_name"], bowling["player_name"]]))})
    player_metadata = players.merge(batting, on="player_name", how="left").merge(bowling, on="player_name", how="left").merge(
        seed, on="player_name", how="left"
    )
    player_metadata[["total_runs", "balls_faced", "credited_wickets", "balls_bowled", "runs_conceded"]] = player_metadata[
        ["total_runs", "balls_faced", "credited_wickets", "balls_bowled", "runs_conceded"]
    ].fillna(0)

    inferred_role = np.where(
        player_metadata["role"].notna(),
        player_metadata["role"],
        np.where(
            player_metadata["credited_wickets"] >= 50,
            np.where(player_metadata["total_runs"] >= 1200, "all_rounder", "bowler"),
            np.where(player_metadata["total_runs"] >= 1500, "batter", "utility"),
        ),
    )
    inferred_skill = np.where(
        player_metadata["primary_skill"].notna(),
        player_metadata["primary_skill"],
        np.where(
            player_metadata["credited_wickets"] >= 50,
            np.where(player_metadata["balls_bowled"] > 0, "bowling_core", "utility"),
            np.where(player_metadata["total_runs"] >= 1500, "batting_core", "utility"),
        ),
    )
    player_metadata["role"] = inferred_role
    player_metadata["primary_skill"] = inferred_skill
    player_metadata["bowling_style"] = player_metadata["bowling_style"].fillna("unknown")
    player_metadata["batting_style"] = player_metadata["batting_style"].fillna("unknown")
    return player_metadata.sort_values("player_name").reset_index(drop=True)


def build_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    batter_out = (
        (df["wicket_player_out"].fillna("") == df["batter"])
        & (df["bowler_wicket"] == 1)
    ).astype(int)
    frame = (
        df.assign(boundary=(df["runs_batter"] == 4).astype(int), six=(df["runs_batter"] == 6).astype(int), batter_out=batter_out)
        .groupby(["batter", "bowler"], as_index=False)
        .agg(
            runs=("runs_batter", "sum"),
            balls=("legal_delivery", "sum"),
            dismissals=("batter_out", "sum"),
            boundaries=("boundary", "sum"),
            sixes=("six", "sum"),
        )
    )
    frame["strike_rate"] = np.where(frame["balls"] > 0, frame["runs"] * 100 / frame["balls"], 0.0)
    return frame.sort_values(["dismissals", "runs"], ascending=[False, False]).reset_index(drop=True)


def build_partnerships(df: pd.DataFrame) -> pd.DataFrame:
    partnership_frame = df[["match_id", "innings", "batter", "non_striker", "runs_total", "legal_delivery"]].copy()
    partnership_frame["pair_a"] = partnership_frame[["batter", "non_striker"]].min(axis=1)
    partnership_frame["pair_b"] = partnership_frame[["batter", "non_striker"]].max(axis=1)
    pair_summary = (
        partnership_frame.groupby(["pair_a", "pair_b"], as_index=False)
        .agg(
            runs=("runs_total", "sum"),
            balls=("legal_delivery", "sum"),
            shared_innings=("match_id", "nunique"),
        )
    )
    pair_summary["run_rate"] = np.where(pair_summary["balls"] > 0, pair_summary["runs"] * 6 / pair_summary["balls"], 0.0)
    pair_summary["average_partnership"] = np.where(
        pair_summary["shared_innings"] > 0,
        pair_summary["runs"] / pair_summary["shared_innings"],
        0.0,
    )

    batter_baseline = (
        df.groupby("batter", as_index=False)
        .agg(batter_runs=("runs_total", "sum"), batter_balls=("legal_delivery", "sum"))
        .rename(columns={"batter": "player_name"})
    )
    batter_baseline["baseline_run_rate"] = np.where(
        batter_baseline["batter_balls"] > 0,
        batter_baseline["batter_runs"] * 6 / batter_baseline["batter_balls"],
        0.0,
    )
    pair_summary = pair_summary.merge(
        batter_baseline[["player_name", "baseline_run_rate"]],
        left_on="pair_a",
        right_on="player_name",
        how="left",
    ).rename(columns={"baseline_run_rate": "pair_a_baseline"}).drop(columns=["player_name"])
    pair_summary = pair_summary.merge(
        batter_baseline[["player_name", "baseline_run_rate"]],
        left_on="pair_b",
        right_on="player_name",
        how="left",
    ).rename(columns={"baseline_run_rate": "pair_b_baseline"}).drop(columns=["player_name"])
    pair_summary["synergy_index"] = pair_summary["run_rate"] - (
        (pair_summary["pair_a_baseline"].fillna(0) + pair_summary["pair_b_baseline"].fillna(0)) / 2
    )
    return pair_summary.sort_values("runs", ascending=False).reset_index(drop=True)


def build_pressure_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pressure_df = df[df["is_pressure"] == 1].copy()
    batter_match_eligibility = (
        df.groupby("batter", as_index=False)
        .agg(matches_played=("match_id", "nunique"))
        .rename(columns={"batter": "player_name"})
    )
    bowler_match_eligibility = (
        df.groupby("bowler", as_index=False)
        .agg(matches_played=("match_id", "nunique"))
        .rename(columns={"bowler": "player_name"})
    )

    batter_table = (
        pressure_df.assign(boundary=(pressure_df["runs_batter"] == 4).astype(int), six=(pressure_df["runs_batter"] == 6).astype(int))
        .groupby("batter", as_index=False)
        .agg(
            innings=("match_id", "nunique"),
            runs=("runs_batter", "sum"),
            balls=("legal_delivery", "sum"),
            boundaries=("boundary", "sum"),
            sixes=("six", "sum"),
        )
        .rename(columns={"batter": "player_name"})
    )
    batter_table = batter_table.merge(batter_match_eligibility, on="player_name", how="left")
    batter_table["strike_rate"] = np.where(batter_table["balls"] > 0, batter_table["runs"] * 100 / batter_table["balls"], 0.0)
    batter_table["pressure_index"] = np.where(
        batter_table["innings"] > 0,
        (batter_table["runs"] / batter_table["innings"]) * (batter_table["strike_rate"] / 100),
        0.0,
    )
    batter_table = batter_table[batter_table["innings"] >= MIN_PRI_INNINGS].copy()

    batter_out = (
        (pressure_df["wicket_player_out"].fillna("") == pressure_df["batter"])
        & (pressure_df["bowler_wicket"] == 1)
    ).astype(int)
    bowler_table = (
        pressure_df.assign(batter_out=batter_out)
        .groupby("bowler", as_index=False)
        .agg(
            innings=("match_id", "nunique"),
            wickets=("batter_out", "sum"),
            balls=("legal_delivery", "sum"),
            runs_conceded=("runs_total", "sum"),
        )
        .rename(columns={"bowler": "player_name"})
    )
    bowler_table = bowler_table.merge(bowler_match_eligibility, on="player_name", how="left")
    bowler_table["economy"] = np.where(bowler_table["balls"] > 0, bowler_table["runs_conceded"] * 6 / bowler_table["balls"], 0.0)
    bowler_table["pressure_index"] = np.where(
        bowler_table["balls"] > 0,
        (bowler_table["wickets"] * 100 / bowler_table["balls"]) * 8 - bowler_table["economy"],
        0.0,
    )
    bowler_table = bowler_table[bowler_table["innings"] >= MIN_PRI_INNINGS].copy()
    return batter_table.sort_values("pressure_index", ascending=False), bowler_table.sort_values(
        "pressure_index", ascending=False
    )


def build_dashboard_summary(matches: pd.DataFrame, df: pd.DataFrame) -> dict:
    decided_matches = matches[matches["winner"] != ""].copy()
    toss_total = max(len(decided_matches), 1)
    toss_winner_rate = round(decided_matches["toss_winner_won"].sum() * 100 / toss_total, 2)
    toss_loser_rate = round(decided_matches["toss_loser_won"].sum() * 100 / toss_total, 2)

    phase_table = (
        df[df["winner"] != ""]
        .groupby(["phase", "batting_team_won"], as_index=False)
        .agg(total_runs=("runs_total", "sum"), matches=("match_id", "nunique"))
    )
    phase_points = []
    for phase in ["powerplay", "middle", "death"]:
        winners = phase_table[(phase_table["phase"] == phase) & (phase_table["batting_team_won"] == 1)]
        losers = phase_table[(phase_table["phase"] == phase) & (phase_table["batting_team_won"] == 0)]
        phase_points.append(
            {
                "phase": phase,
                "winners": round(float(winners["total_runs"].sum() / max(winners["matches"].sum(), 1)), 2),
                "losers": round(float(losers["total_runs"].sum() / max(losers["matches"].sum(), 1)), 2),
            }
        )

    batter_board = (
        df.groupby("batter", as_index=False)["runs_batter"].sum().sort_values("runs_batter", ascending=False).head(5)
    )
    bowler_board = (
        df.groupby("bowler", as_index=False)["bowler_wicket"].sum().sort_values("bowler_wicket", ascending=False).head(5)
    )
    top_batters = [
        {"rank": index + 1, "name": row["batter"], "value": int(row["runs_batter"])}
        for index, (_, row) in enumerate(batter_board.iterrows())
    ]
    top_bowlers = [
        {"rank": index + 1, "name": row["bowler"], "value": int(row["bowler_wicket"])}
        for index, (_, row) in enumerate(bowler_board.iterrows())
    ]

    innings_totals = (
        df.groupby(["match_id", "innings"], as_index=False)
        .agg(batting_team=("batting_team", "first"), innings_runs=("runs_total", "sum"))
    )
    first_innings = innings_totals[innings_totals["innings"] == 1][["match_id", "batting_team", "innings_runs"]].rename(
        columns={"batting_team": "first_innings_team", "innings_runs": "first_innings_runs"}
    )
    surprise_table = first_innings.merge(
        matches[["match_id", "winner", "second_innings_team", "target_score"]], on="match_id", how="left"
    )
    lost_250_plus = int(
        (
            (surprise_table["first_innings_runs"] >= 250)
            & (surprise_table["winner"] != surprise_table["first_innings_team"])
        ).sum()
    )
    highest_successful_chase = int(
        surprise_table.loc[surprise_table["winner"] == surprise_table["second_innings_team"], "target_score"].max()
    )
    highest_losing_first_innings = int(
        surprise_table.loc[
            surprise_table["winner"] != surprise_table["first_innings_team"], "first_innings_runs"
        ].max()
    )
    surprising_finding = (
        f"A first-innings score of {highest_losing_first_innings} still was not safe in this dataset, "
        f"because teams lost {lost_250_plus} matches after scoring 250-plus and the highest successful chase reached "
        f"{highest_successful_chase}."
    )

    team_match_rows = pd.concat(
        [
            decided_matches.assign(team=decided_matches["team1"]),
            decided_matches.assign(team=decided_matches["team2"]),
        ],
        ignore_index=True,
    )
    team_match_rows = team_match_rows[team_match_rows["team"] != ""].copy()
    team_match_rows["won_match"] = (team_match_rows["winner"] == team_match_rows["team"]).astype(int)
    team_match_rows["won_toss"] = (team_match_rows["toss_winner"] == team_match_rows["team"]).astype(int)
    team_match_rows["lost_toss"] = 1 - team_match_rows["won_toss"]
    team_match_rows["batting_first"] = (team_match_rows["first_innings_team"] == team_match_rows["team"]).astype(int)
    team_match_rows["bowling_first"] = (team_match_rows["second_innings_team"] == team_match_rows["team"]).astype(int)
    team_match_rows["toss_won_win"] = team_match_rows["won_match"] * team_match_rows["won_toss"]
    team_match_rows["toss_lost_win"] = team_match_rows["won_match"] * team_match_rows["lost_toss"]
    team_match_rows["batting_first_win"] = team_match_rows["won_match"] * team_match_rows["batting_first"]
    team_match_rows["bowling_first_win"] = team_match_rows["won_match"] * team_match_rows["bowling_first"]
    team_match_rows["eligible_chasing_200_plus"] = (
        (team_match_rows["second_innings_team"] == team_match_rows["team"])
        & (team_match_rows["target_score"] >= 200)
    ).astype(int)
    team_match_rows["successful_chasing_200_plus"] = (
        (team_match_rows["eligible_chasing_200_plus"] == 1)
        & (team_match_rows["won_match"] == 1)
    ).astype(int)
    team_match_rows["eligible_defending_150_or_lower"] = (
        (team_match_rows["first_innings_team"] == team_match_rows["team"])
        & (team_match_rows["first_innings_runs"] <= 150)
    ).astype(int)
    team_match_rows["successful_defending_150_or_lower"] = (
        (team_match_rows["eligible_defending_150_or_lower"] == 1)
        & (team_match_rows["won_match"] == 1)
    ).astype(int)

    team_summary = (
        team_match_rows.groupby("team", as_index=False)
        .agg(
            toss_won_matches=("won_toss", "sum"),
            toss_won_wins=("toss_won_win", "sum"),
            toss_lost_matches=("lost_toss", "sum"),
            toss_lost_wins=("toss_lost_win", "sum"),
            batting_first_matches=("batting_first", "sum"),
            batting_first_wins=("batting_first_win", "sum"),
            bowling_first_matches=("bowling_first", "sum"),
            bowling_first_wins=("bowling_first_win", "sum"),
            chasing_200_plus_matches=("eligible_chasing_200_plus", "sum"),
            chasing_200_plus_wins=("successful_chasing_200_plus", "sum"),
            defending_150_or_lower_matches=("eligible_defending_150_or_lower", "sum"),
            defending_150_or_lower_wins=("successful_defending_150_or_lower", "sum"),
        )
        .sort_values("team")
        .reset_index(drop=True)
    )

    team_toss_impact = [
        {
            "team": row["team"],
            "toss_won_matches": int(row["toss_won_matches"]),
            "toss_lost_matches": int(row["toss_lost_matches"]),
            "win_pct_after_winning_toss": calculate_win_percentage(row["toss_won_wins"], row["toss_won_matches"]),
            "win_pct_after_losing_toss": calculate_win_percentage(row["toss_lost_wins"], row["toss_lost_matches"]),
        }
        for _, row in team_summary.iterrows()
    ]
    team_innings_impact = [
        {
            "team": row["team"],
            "batting_first_matches": int(row["batting_first_matches"]),
            "bowling_first_matches": int(row["bowling_first_matches"]),
            "win_pct_batting_first": calculate_win_percentage(row["batting_first_wins"], row["batting_first_matches"]),
            "win_pct_bowling_first": calculate_win_percentage(row["bowling_first_wins"], row["bowling_first_matches"]),
        }
        for _, row in team_summary.iterrows()
    ]
    team_milestone_impact = [
        {
            "team": row["team"],
            "chasing_200_plus_matches": int(row["chasing_200_plus_matches"]),
            "defending_150_or_lower_matches": int(row["defending_150_or_lower_matches"]),
            "win_pct_chasing_200_plus": calculate_win_percentage(row["chasing_200_plus_wins"], row["chasing_200_plus_matches"]),
            "win_pct_defending_150_or_lower": calculate_win_percentage(
                row["defending_150_or_lower_wins"], row["defending_150_or_lower_matches"]
            ),
        }
        for _, row in team_summary.iterrows()
    ]

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "scope": "All seasons in workspace dataset",
        "toss_impact": [
            {"outcome": "Won toss", "win_rate": toss_winner_rate, "matches": int(toss_total)},
            {"outcome": "Lost toss", "win_rate": toss_loser_rate, "matches": int(toss_total)},
        ],
        "team_toss_impact": team_toss_impact,
        "team_innings_impact": team_innings_impact,
        "team_milestone_impact": team_milestone_impact,
        "phase_comparison": phase_points,
        "top_batters": top_batters,
        "top_bowlers": top_bowlers,
        "surprising_finding": surprising_finding,
    }


def build_audit_summary(df: pd.DataFrame, matches: pd.DataFrame) -> dict:
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_file": str(RAW_DATA_PATH.relative_to(ROOT_DIR)),
        "cleaning_script": "ml/scripts/clean_ipl_data.py",
        "rows_in_raw": int(len(df)),
        "matches_in_dataset": int(matches["match_id"].nunique()),
        "seasons": sorted(df["season_label"].unique().tolist()),
        "mappings": {
            "team_aliases": TEAM_ALIASES,
            "venue_aliases": VENUE_ALIASES,
            "city_aliases": CITY_ALIASES,
        },
        "derived_fields": [
            "season_year",
            "season_label",
            "legal_delivery",
            "wicket_fell",
            "bowler_wicket",
            "runs_before",
            "runs_after",
            "legal_balls_before",
            "wickets_before",
            "current_rr_before",
            "wickets_in_hand_before",
            "phase",
            "target_score",
            "required_runs_before",
            "required_rr_before",
            "fielding_team",
            "pressure_state",
        ],
    }


def write_artifacts(artifacts: PipelineArtifacts) -> None:
    artifacts.deliveries.to_parquet(PROCESSED_DIR / "deliveries.parquet", index=False)
    artifacts.matches.to_parquet(PROCESSED_DIR / "matches.parquet", index=False)
    artifacts.head_to_head.to_parquet(PROCESSED_DIR / "head_to_head.parquet", index=False)
    artifacts.partnerships.to_parquet(PROCESSED_DIR / "partnerships.parquet", index=False)
    artifacts.pressure_batting.to_parquet(PROCESSED_DIR / "pressure_batting.parquet", index=False)
    artifacts.pressure_bowling.to_parquet(PROCESSED_DIR / "pressure_bowling.parquet", index=False)
    artifacts.player_metadata.to_parquet(PROCESSED_DIR / "player_metadata.parquet", index=False)
    (REPORTS_DIR / "dashboard_summary.json").write_text(json.dumps(artifacts.dashboard_summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "data_audit_summary.json").write_text(json.dumps(artifacts.audit_summary, indent=2), encoding="utf-8")


def run_pipeline() -> PipelineArtifacts:
    ensure_directories()
    deliveries = enrich_match_state(load_raw_data())
    matches = build_match_table(deliveries)
    head_to_head = build_head_to_head(deliveries)
    partnerships = build_partnerships(deliveries)
    pressure_batting, pressure_bowling = build_pressure_tables(deliveries)
    player_metadata = build_player_metadata(deliveries)
    dashboard_summary = build_dashboard_summary(matches, deliveries)
    audit_summary = build_audit_summary(deliveries, matches)
    artifacts = PipelineArtifacts(
        deliveries=deliveries,
        matches=matches,
        head_to_head=head_to_head,
        partnerships=partnerships,
        pressure_batting=pressure_batting,
        pressure_bowling=pressure_bowling,
        player_metadata=player_metadata,
        dashboard_summary=dashboard_summary,
        audit_summary=audit_summary,
    )
    write_artifacts(artifacts)
    return artifacts


if __name__ == "__main__":
    run_pipeline()
