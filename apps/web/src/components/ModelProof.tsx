import { useEffect, useState } from "react";

import { api, chartUrl } from "../lib/api";
import type { InsightsResponse } from "../types/api";

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function ModelProof() {
  const [insights, setInsights] = useState<InsightsResponse | null>(null);

  useEffect(() => {
    api.getInsights().then(setInsights).catch(() => setInsights(null));
  }, []);

  if (!insights) {
    return null;
  }

  const { chase_models: chase, score_models: score } = insights;

  return (
    <>
      <section className="two-column">
        <article className="card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Model Proof</p>
              <h3>Chase model leaderboard</h3>
            </div>
            <span className="scope-pill">Tested on {chase.holdout_seasons.join(", ")}</span>
          </div>
          <p className="muted-copy">Every model gets the same inputs: {chase.features.join(", ")}.</p>
          <table className="stat-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Accuracy</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {chase.leaderboard.map((row, index) => (
                <tr key={row.model}>
                  <td>{index + 1}</td>
                  <td>{row.model}</td>
                  <td>{formatPercent(row.accuracy)}</td>
                  <td>{row.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>

        <article className="card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Hyperparameter Tuning</p>
              <h3>Before vs after tuning</h3>
            </div>
          </div>
          <table className="stat-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Search</th>
                <th>AUC before</th>
                <th>AUC after</th>
              </tr>
            </thead>
            <tbody>
              {chase.tuning.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.method}</td>
                  <td>{row.before.roc_auc.toFixed(3)}</td>
                  <td>{row.after.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="insight-list">
            {chase.tuning.map((row) => (
              <li key={row.model}>
                {row.model} best settings:{" "}
                {Object.entries(row.best_params)
                  .map(([key, value]) => `${key} = ${value}`)
                  .join(", ")}
              </li>
            ))}
          </ul>
        </article>
      </section>

      <article className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Score Projection Models</p>
            <h3>How close are the final-score predictions?</h3>
          </div>
        </div>
        <table className="stat-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>R²</th>
              <th>MAE (runs)</th>
              <th>RMSE (runs)</th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: "Simple Linear (runs at 10 overs)", metrics: score.simple },
              { name: "Multiple Linear", metrics: score.multiple },
              { name: "Support Vector Regression", metrics: score.svr },
            ].map(({ name, metrics }) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{metrics.r2.toFixed(3)}</td>
                <td>{metrics.mae.toFixed(1)}</td>
                <td>{metrics.rmse.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="footnote">
          The simple model is checked at the 10-over mark only. The multiple and SVR models are checked at every ball of the innings, which is harder.
        </p>
      </article>

      <article className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Explainable Rules</p>
            <h3>Decision tree for chases</h3>
          </div>
        </div>
        {chase.tree_chart ? (
          <a href={chartUrl(chase.tree_chart)} target="_blank" rel="noreferrer" className="tree-chart">
            <img src={chartUrl(chase.tree_chart)} alt="Decision tree for chases" />
          </a>
        ) : null}
        <p className="footnote">
          Start at the top and follow True (left) or False (right). Blue boxes favour the chasing side, orange boxes favour the defending side, and
          darker means more certain. Click the tree to open it full size.
        </p>
        <details>
          <summary className="muted-copy">Show the same rules as text</summary>
          <pre className="rules-block">{chase.tree_rules}</pre>
        </details>
      </article>
    </>
  );
}
