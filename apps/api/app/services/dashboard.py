from __future__ import annotations

from ..data_access import AnalyticsRepository
from ..models import DashboardSummaryResponse, DataProofResponse, OptionItem, OptionsResponse, TeamPlayerPool


def get_dashboard_summary(repository: AnalyticsRepository) -> DashboardSummaryResponse:
    return DashboardSummaryResponse.model_validate(repository.read_json_report("dashboard_summary.json"))


def get_data_proof(repository: AnalyticsRepository) -> DataProofResponse:
    return DataProofResponse.model_validate(repository.read_json_report("data_audit_summary.json"))


def get_options(repository: AnalyticsRepository) -> OptionsResponse:
    batters = repository.fetch_dataframe(
        "SELECT DISTINCT batter AS value FROM deliveries WHERE batter <> '' ORDER BY batter"
    )["value"].tolist()
    bowlers = repository.fetch_dataframe(
        "SELECT DISTINCT bowler AS value FROM deliveries WHERE bowler <> '' ORDER BY bowler"
    )["value"].tolist()
    teams = repository.fetch_dataframe(
        "SELECT DISTINCT team1 AS value FROM matches UNION SELECT DISTINCT team2 AS value FROM matches ORDER BY value"
    )["value"].tolist()
    venues = repository.fetch_dataframe(
        "SELECT DISTINCT venue AS value FROM matches WHERE venue <> '' ORDER BY venue"
    )["value"].tolist()
    captain_batters = repository.fetch_dataframe(
        """
        SELECT batting_team AS team, batter AS player, COUNT(DISTINCT match_id) AS matches
        FROM deliveries
        WHERE batting_team <> '' AND batter <> ''
        GROUP BY batting_team, batter
        ORDER BY batting_team, matches DESC, batter
        """
    )
    captain_bowlers = repository.fetch_dataframe(
        """
        SELECT fielding_team AS team, bowler AS player, COUNT(DISTINCT match_id) AS matches
        FROM deliveries
        WHERE fielding_team <> '' AND bowler <> ''
        GROUP BY fielding_team, bowler
        ORDER BY fielding_team, matches DESC, bowler
        """
    )

    batter_pools = [
        TeamPlayerPool(
            team=team,
            players=[OptionItem(label=player, value=player) for player in group["player"].tolist()],
        )
        for team, group in captain_batters.groupby("team", sort=True)
    ]
    bowler_pools = [
        TeamPlayerPool(
            team=team,
            players=[OptionItem(label=player, value=player) for player in group["player"].tolist()],
        )
        for team, group in captain_bowlers.groupby("team", sort=True)
    ]
    return OptionsResponse(
        batters=[OptionItem(label=item, value=item) for item in batters],
        bowlers=[OptionItem(label=item, value=item) for item in bowlers],
        teams=[OptionItem(label=item, value=item) for item in teams],
        venues=[OptionItem(label=item, value=item) for item in venues],
        captain_batters_by_team=batter_pools,
        captain_bowlers_by_team=bowler_pools,
    )
