import { FormEvent, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { ImpactOptimizationRequest, ImpactOptimizationResponse, OptionsResponse } from "../types/api";

export function ImpactLabPage() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [result, setResult] = useState<ImpactOptimizationResponse | null>(null);
  const [form, setForm] = useState<ImpactOptimizationRequest>({
    innings: 2,
    current_score: 94,
    wickets_lost: 4,
    overs_completed: 11,
    balls_into_over: 3,
    target_score: 176,
    batting_team: "",
    bowling_team: "",
    venue: "",
    player_role: "finisher",
  });

  useEffect(() => {
    api.getOptions().then((payload) => {
      setOptions(payload);
      setForm((current) => ({
        ...current,
        batting_team: payload.teams[0]?.value ?? "",
        bowling_team: payload.teams[1]?.value ?? payload.teams[0]?.value ?? "",
        venue: payload.venues[0]?.value ?? "",
      }));
    });
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setResult(await api.impactOptimize(form));
  };

  return (
    <section className="two-column">
      <form className="card form-card" onSubmit={submit}>
        <div className="section-head">
          <div>
            <p className="eyebrow">Impact Player Optimizer</p>
            <h2>Best over band for maximum swing</h2>
          </div>
        </div>
        <div className="form-grid">
          <label>
            Role
            <select value={form.player_role} onChange={(event) => setForm({ ...form, player_role: event.target.value as ImpactOptimizationRequest["player_role"] })}>
              <option value="finisher">Finisher</option>
              <option value="anchor">Anchor</option>
              <option value="spinner">Spinner</option>
              <option value="pacer">Pacer</option>
              <option value="all_rounder">All-rounder</option>
            </select>
          </label>
          <label>
            Batting team
            <select value={form.batting_team} onChange={(event) => setForm({ ...form, batting_team: event.target.value })}>
              {options?.teams.map((team) => (
                <option key={team.value} value={team.value}>
                  {team.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Bowling team
            <select value={form.bowling_team} onChange={(event) => setForm({ ...form, bowling_team: event.target.value })}>
              {options?.teams.map((team) => (
                <option key={team.value} value={team.value}>
                  {team.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Venue
            <select value={form.venue} onChange={(event) => setForm({ ...form, venue: event.target.value })}>
              {options?.venues.map((venue) => (
                <option key={venue.value} value={venue.value}>
                  {venue.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Score
            <input type="number" value={form.current_score} onChange={(event) => setForm({ ...form, current_score: Number(event.target.value) })} />
          </label>
          <label>
            Wickets
            <input type="number" value={form.wickets_lost} onChange={(event) => setForm({ ...form, wickets_lost: Number(event.target.value) })} />
          </label>
          <label>
            Over
            <input type="number" value={form.overs_completed} onChange={(event) => setForm({ ...form, overs_completed: Number(event.target.value) })} />
          </label>
          <label>
            Ball
            <input type="number" value={form.balls_into_over} onChange={(event) => setForm({ ...form, balls_into_over: Number(event.target.value) })} />
          </label>
        </div>
        <button type="submit">Simulate impact timing</button>
      </form>

      <section className="card result-card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Recommendation</p>
            <h2>{result ? result.recommended_over_band : "No simulation yet"}</h2>
          </div>
        </div>
        {result ? (
          <>
            <p className="result-highlight">Best projected gain: +{result.best_gain}% win probability.</p>
            <div className="impact-list">
              {result.scenarios.map((scenario) => (
                <article key={scenario.over_band} className="impact-card">
                  <strong>{scenario.over_band}</strong>
                  <span>{scenario.estimated_win_probability}% projected win probability</span>
                  <p>{scenario.reason}</p>
                </article>
              ))}
            </div>
            <p className="footnote">{result.assumption_note}</p>
          </>
        ) : (
          <p className="muted-copy">Run a simulation to find the over window where a role-specific sub creates the biggest marginal lift.</p>
        )}
      </section>
    </section>
  );
}

