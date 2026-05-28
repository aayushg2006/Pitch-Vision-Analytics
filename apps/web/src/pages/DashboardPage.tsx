import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../lib/api";
import type {
  DashboardSummaryResponse,
  TeamInningsImpactDatum,
  TeamMilestoneImpactDatum,
  TeamTossImpactDatum,
} from "../types/api";

type TeamMetricDatum = TeamTossImpactDatum | TeamInningsImpactDatum | TeamMilestoneImpactDatum;

type TeamMetricChartProps = {
  title: string;
  eyebrow: string;
  data: TeamMetricDatum[];
  bars: Array<{
    key: string;
    name: string;
    color: string;
    matchKey: string;
  }>;
};

function TeamMetricTooltip({
  active,
  payload,
  label,
  bars,
}: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; payload: Record<string, number | string> }>;
  label?: string;
  bars: TeamMetricChartProps["bars"];
}) {
  if (!active || !payload?.length) {
    return null;
  }

  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      {payload.map((entry) => {
        const barMeta = bars.find((bar) => bar.name === entry.name);
        const matches = barMeta ? entry.payload[String(barMeta.matchKey)] : undefined;
        return (
          <div key={entry.name} className="tooltip-row">
            <span>{entry.name}</span>
            <strong>{entry.value}%</strong>
            {typeof matches === "number" ? <small>{matches} matches</small> : null}
          </div>
        );
      })}
    </div>
  );
}

function TeamMetricChart({ title, eyebrow, data, bars }: TeamMetricChartProps) {
  const chartHeight = Math.max(460, data.length * 42);

  return (
    <article className="card chart-card tall-chart-card">
      <div className="section-head">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
        </div>
      </div>
      <div className="team-chart-scroll">
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 18, left: 12, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#20314e" />
            <XAxis type="number" stroke="#b9c7dd" domain={[0, 100]} />
            <YAxis type="category" dataKey="team" stroke="#b9c7dd" width={176} interval={0} />
            <Tooltip content={<TeamMetricTooltip bars={bars} />} />
            <Legend wrapperStyle={{ color: "#f4f7fb" }} />
            {bars.map((bar) => (
              <Bar key={String(bar.key)} dataKey={String(bar.key)} name={bar.name} fill={bar.color} radius={[0, 8, 8, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

export function DashboardPage() {
  const [data, setData] = useState<DashboardSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDashboardSummary().then(setData).catch((reason) => setError(String(reason)));
  }, []);

  if (error) {
    return <section className="card error-card">Unable to load dashboard: {error}</section>;
  }

  if (!data) {
    return <section className="card">Loading dashboard intelligence...</section>;
  }

  return (
    <>
      <section className="card finding-card">
        <p className="eyebrow">Surprising Finding</p>
        <h2>{data.surprising_finding}</h2>
      </section>

      <section className="chart-grid">
        <TeamMetricChart
          title="Toss Impact Analyzer"
          eyebrow="Mandatory Insight"
          data={data.team_toss_impact}
          bars={[
            {
              key: "win_pct_after_winning_toss",
              name: "Won toss",
              color: "#f6b94d",
              matchKey: "toss_won_matches",
            },
            {
              key: "win_pct_after_losing_toss",
              name: "Lost toss",
              color: "#1ec8a5",
              matchKey: "toss_lost_matches",
            },
          ]}
        />

        <TeamMetricChart
          title="Bat First vs Bowl First"
          eyebrow="Team Strategy Split"
          data={data.team_innings_impact}
          bars={[
            {
              key: "win_pct_batting_first",
              name: "Batting first",
              color: "#f6b94d",
              matchKey: "batting_first_matches",
            },
            {
              key: "win_pct_bowling_first",
              name: "Bowling first",
              color: "#5cbcff",
              matchKey: "bowling_first_matches",
            },
          ]}
        />

        <TeamMetricChart
          title="Pressure Threshold Wins"
          eyebrow="Milestone Scenarios"
          data={data.team_milestone_impact}
          bars={[
            {
              key: "win_pct_chasing_200_plus",
              name: "Chasing 200+",
              color: "#ff8c42",
              matchKey: "chasing_200_plus_matches",
            },
            {
              key: "win_pct_defending_150_or_lower",
              name: "Defending 150 or lower",
              color: "#7ce38b",
              matchKey: "defending_150_or_lower_matches",
            },
          ]}
        />

        <article className="card chart-card compact-chart-card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Mandatory Insight</p>
              <h3>Phase Analyzer</h3>
            </div>
            <span className="scope-pill">{data.scope}</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.phase_comparison}>
              <CartesianGrid strokeDasharray="3 3" stroke="#20314e" />
              <XAxis dataKey="phase" stroke="#b9c7dd" />
              <YAxis stroke="#b9c7dd" />
              <Tooltip />
              <Bar dataKey="winners" name="Winning teams" fill="#f6b94d" radius={[8, 8, 0, 0]} />
              <Bar dataKey="losers" name="Losing teams" fill="#1ec8a5" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </section>

      <section className="table-grid">
        <article className="card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Elite Leaderboard</p>
              <h3>Top 5 Batters</h3>
            </div>
          </div>
          <table className="stat-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Runs</th>
              </tr>
            </thead>
            <tbody>
              {data.top_batters.map((entry) => (
                <tr key={entry.name}>
                  <td>{entry.rank}</td>
                  <td>{entry.name}</td>
                  <td>{entry.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>

        <article className="card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Elite Leaderboard</p>
              <h3>Top 5 Bowlers</h3>
            </div>
          </div>
          <table className="stat-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Wickets</th>
              </tr>
            </thead>
            <tbody>
              {data.top_bowlers.map((entry) => (
                <tr key={entry.name}>
                  <td>{entry.rank}</td>
                  <td>{entry.name}</td>
                  <td>{entry.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>
      </section>
    </>
  );
}
