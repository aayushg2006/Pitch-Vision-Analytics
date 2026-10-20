import { useEffect, useState } from "react";

import { api, chartUrl } from "../lib/api";
import type { InsightsResponse } from "../types/api";

export function VisualInsights() {
  const [insights, setInsights] = useState<InsightsResponse | null>(null);

  useEffect(() => {
    api.getInsights().then(setInsights).catch(() => setInsights(null));
  }, []);

  if (!insights) {
    return null;
  }

  return (
    <>
      <section className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Visual Insights</p>
            <h3>What the ball-by-ball data looks like</h3>
          </div>
        </div>
        <div className="image-grid">
          {insights.charts.map((chart) => (
            <figure key={chart.name} className="image-card">
              <img src={chartUrl(chart.name)} alt={chart.title} loading="lazy" />
              <figcaption>{chart.description}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Freak Innings Detector</p>
            <h3>The most unusual innings in IPL history</h3>
          </div>
        </div>
        <p className="muted-copy">
          Every innings is scored on runs, wickets, run rate, fours, sixes and dot balls. The innings that sit furthest from normal are flagged.
        </p>
        <table className="stat-table">
          <thead>
            <tr>
              <th>Batting</th>
              <th>Against</th>
              <th>Season</th>
              <th>Score</th>
              <th>Run rate</th>
              <th>4s / 6s</th>
            </tr>
          </thead>
          <tbody>
            {insights.freak_innings.map((row) => (
              <tr key={`${row.batting_team}-${row.season}-${row.score}-${row.innings}`}>
                <td>{row.batting_team}</td>
                <td>{row.fielding_team}</td>
                <td>{row.season}</td>
                <td>{row.score}</td>
                <td>{row.run_rate.toFixed(2)}</td>
                <td>
                  {row.fours} / {row.sixes}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
