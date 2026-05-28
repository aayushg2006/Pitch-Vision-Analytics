from __future__ import annotations

from ..data_access import AnalyticsRepository
from ..models import (
    HeadToHeadRequest,
    HeadToHeadResponse,
    PartnershipRequest,
    PartnershipResponse,
    PressureFilter,
    PressureOverviewResponse,
    PressurePlayerRow,
)


def get_head_to_head(repository: AnalyticsRepository, payload: HeadToHeadRequest) -> HeadToHeadResponse:
    frame = repository.fetch_dataframe(
        """
        SELECT batter, bowler, runs, balls, dismissals, strike_rate, boundaries, sixes
        FROM head_to_head
        WHERE batter = ? AND bowler = ?
        LIMIT 1
        """,
        [payload.batter, payload.bowler],
    )
    if frame.empty:
        return HeadToHeadResponse(
            batter=payload.batter,
            bowler=payload.bowler,
            runs=0,
            balls=0,
            dismissals=0,
            strike_rate=0.0,
            boundaries=0,
            sixes=0,
        )
    row = frame.iloc[0].to_dict()
    return HeadToHeadResponse.model_validate(row)


def get_partnership(repository: AnalyticsRepository, payload: PartnershipRequest) -> PartnershipResponse:
    pair_a, pair_b = sorted([payload.batter_one, payload.batter_two])
    frame = repository.fetch_dataframe(
        """
        SELECT pair_a, pair_b, runs, balls, run_rate, average_partnership, shared_innings, synergy_index
        FROM partnerships
        WHERE pair_a = ? AND pair_b = ?
        LIMIT 1
        """,
        [pair_a, pair_b],
    )
    if frame.empty:
        return PartnershipResponse(
            pair=(pair_a, pair_b),
            runs=0,
            balls=0,
            run_rate=0.0,
            average_partnership=0.0,
            shared_innings=0,
            synergy_index=0.0,
        )
    row = frame.iloc[0].to_dict()
    return PartnershipResponse(
        pair=(row["pair_a"], row["pair_b"]),
        runs=int(row["runs"]),
        balls=int(row["balls"]),
        run_rate=float(row["run_rate"]),
        average_partnership=float(row["average_partnership"]),
        shared_innings=int(row["shared_innings"]),
        synergy_index=float(row["synergy_index"]),
    )


def get_pressure_stats(repository: AnalyticsRepository, payload: PressureFilter) -> PressureOverviewResponse:
    table_name = "pressure_batting" if payload.role == "batter" else "pressure_bowling"
    if payload.player_name:
        frame = repository.fetch_dataframe(
            f"SELECT * FROM {table_name} WHERE player_name = ? ORDER BY pressure_index DESC LIMIT ?",
            [payload.player_name, payload.limit],
        )
    else:
        frame = repository.fetch_dataframe(
            f"SELECT * FROM {table_name} ORDER BY pressure_index DESC LIMIT ?",
            [payload.limit],
        )
    rows = []
    for row in frame.to_dict(orient="records"):
        if payload.role == "batter":
            rows.append(
                PressurePlayerRow(
                    player_name=row["player_name"],
                    innings=int(row["innings"]),
                    matches_played=int(row["matches_played"]),
                    metric_one=float(row["strike_rate"]),
                    metric_two=float(row["runs"]),
                    pressure_index=float(row["pressure_index"]),
                )
            )
        else:
            rows.append(
                PressurePlayerRow(
                    player_name=row["player_name"],
                    innings=int(row["innings"]),
                    matches_played=int(row["matches_played"]),
                    metric_one=float(row["economy"]),
                    metric_two=float(row["wickets"]),
                    pressure_index=float(row["pressure_index"]),
                )
            )
    return PressureOverviewResponse(role=payload.role, rows=rows)
