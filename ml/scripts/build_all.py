from __future__ import annotations

import json

from clean_ipl_data import run_pipeline
from train_insights import train_insights
from train_models import train_all


def main() -> None:
    pipeline = run_pipeline()
    report = train_all()
    insights = train_insights()
    summary = {
        "rows": len(pipeline.deliveries),
        "matches": int(pipeline.matches["match_id"].nunique()),
        "top_batter": pipeline.dashboard_summary["top_batters"][0]["name"],
        "top_bowler": pipeline.dashboard_summary["top_bowlers"][0]["name"],
        "first_innings_holdouts": report["first_innings"]["holdout_seasons"],
        "second_innings_holdouts": report["second_innings"]["holdout_seasons"],
        "best_chase_model": insights["chase_models"]["leaderboard"][0]["model"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

