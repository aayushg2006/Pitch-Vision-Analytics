import { FormEvent, useState } from "react";

import { api } from "../lib/api";
import type { PressureOverviewResponse } from "../types/api";

export function ClutchModePage() {
  const [role, setRole] = useState<"batter" | "bowler">("batter");
  const [result, setResult] = useState<PressureOverviewResponse | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setResult(await api.clutch({ role, limit: 10 }));
  };

  return (
    <section className="card">
      <div className="section-head">
        <div>
          <p className="eyebrow">Pressure-Resilience Index</p>
          <h2>Clutch mode leaderboard</h2>
        </div>
      </div>
      <p className="muted-copy">Only players with 50 or more pressure innings in the dataset are eligible for the PRI leaderboard.</p>
      <form className="inline-form" onSubmit={submit}>
        <label>
          Role lens
          <select value={role} onChange={(event) => setRole(event.target.value as "batter" | "bowler")}>
            <option value="batter">Batters</option>
            <option value="bowler">Bowlers</option>
          </select>
        </label>
        <button type="submit">Load clutch table</button>
      </form>
      {result ? (
        <table className="stat-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>Matches</th>
              <th>Innings</th>
              <th>{result.role === "batter" ? "Strike rate" : "Economy"}</th>
              <th>{result.role === "batter" ? "Runs" : "Wickets"}</th>
              <th>PRI</th>
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row) => (
              <tr key={row.player_name}>
                <td>{row.player_name}</td>
                <td>{row.matches_played}</td>
                <td>{row.innings}</td>
                <td>{row.metric_one.toFixed(2)}</td>
                <td>{row.metric_two.toFixed(2)}</td>
                <td>{row.pressure_index.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted-copy">Toggle the clutch lens to isolate who still wins the game when the pressure spikes.</p>
      )}
    </section>
  );
}
