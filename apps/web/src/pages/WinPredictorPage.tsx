import { FormEvent, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { OptionsResponse, WinPredictRequest, WinPredictResponse } from "../types/api";

const initialForm: WinPredictRequest = {
  innings: 2,
  current_score: 98,
  wickets_lost: 3,
  overs_completed: 12,
  balls_into_over: 2,
  target_score: 181,
  batting_team: "",
  bowling_team: "",
  venue: "",
};

export function WinPredictorPage() {
  const [form, setForm] = useState<WinPredictRequest>(initialForm);
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [result, setResult] = useState<WinPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    setError(null);
    try {
      const payload = form.innings === 1 ? { ...form, target_score: undefined } : form;
      setResult(await api.predict(payload));
    } catch (reason) {
      setError(String(reason));
    }
  };

  return (
    <section className="two-column">
      <form className="card form-card" onSubmit={submit}>
        <div className="section-head">
          <div>
            <p className="eyebrow">Live Match Win Predictor</p>
            <h2>Calibrated innings-state forecast</h2>
          </div>
        </div>

        <label>
          Innings
          <select value={form.innings} onChange={(event) => setForm({ ...form, innings: Number(event.target.value) as 1 | 2 })}>
            <option value={1}>1st innings</option>
            <option value={2}>2nd innings</option>
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
        <div className="form-grid">
          <label>
            Score
            <input type="number" value={form.current_score} onChange={(event) => setForm({ ...form, current_score: Number(event.target.value) })} />
          </label>
          <label>
            Wickets lost
            <input type="number" min={0} max={10} value={form.wickets_lost} onChange={(event) => setForm({ ...form, wickets_lost: Number(event.target.value) })} />
          </label>
          <label>
            Overs completed
            <input type="number" min={0} max={20} value={form.overs_completed} onChange={(event) => setForm({ ...form, overs_completed: Number(event.target.value) })} />
          </label>
          <label>
            Balls into over
            <input type="number" min={0} max={5} value={form.balls_into_over} onChange={(event) => setForm({ ...form, balls_into_over: Number(event.target.value) })} />
          </label>
          {form.innings === 2 ? (
            <label className="full-width">
              Target score
              <input type="number" min={1} value={form.target_score ?? ""} onChange={(event) => setForm({ ...form, target_score: Number(event.target.value) })} />
            </label>
          ) : null}
        </div>
        <button type="submit">Project win probability</button>
      </form>

      <section className="card result-card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Model Output</p>
            <h2>{result ? `${result.win_probability}%` : "Waiting for match state"}</h2>
          </div>
        </div>
        {error ? <p className="error-copy">{error}</p> : null}
        {result ? (
          <>
            <p className="result-highlight">{result.likely_winner} currently has the stronger win path.</p>
            <div className="mini-stats">
              <div>
                <span>Confidence band</span>
                <strong>
                  {result.confidence_band[0]}% - {result.confidence_band[1]}%
                </strong>
              </div>
              <div>
                <span>Model version</span>
                <strong>{result.model_version}</strong>
              </div>
            </div>
            <ul className="insight-list">
              {result.top_factors.map((factor) => (
                <li key={factor}>{factor}</li>
              ))}
            </ul>
            {result.projected_score ? (
              <div className="sub-panel">
                <p className="eyebrow">Projected Final Score</p>
                <div className="mini-stats">
                  <div>
                    <span>Linear Regression</span>
                    <strong>{Math.round(result.projected_score.linear_regression)}</strong>
                  </div>
                  <div>
                    <span>Support Vector Regression</span>
                    <strong>{Math.round(result.projected_score.svr)}</strong>
                  </div>
                </div>
                <p className="footnote">Rule of thumb: {result.projected_score.rule_of_thumb}</p>
              </div>
            ) : null}
            {result.model_opinions ? (
              <div className="sub-panel">
                <p className="eyebrow">Second Opinions</p>
                <table className="stat-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Chase win chance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.model_opinions.map((opinion) => (
                      <tr key={opinion.model}>
                        <td>{opinion.model}</td>
                        <td>{opinion.probability.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            {result.decision_path ? (
              <div className="sub-panel">
                <p className="eyebrow">Decision Path</p>
                <ol className="insight-list">
                  {result.decision_path.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </div>
            ) : null}
            <p className="footnote">{result.evaluation_stamp}</p>
          </>
        ) : (
          <p className="muted-copy">Feed the live game context and the backend will return a calibrated probability with reason codes.</p>
        )}
      </section>
    </section>
  );
}

