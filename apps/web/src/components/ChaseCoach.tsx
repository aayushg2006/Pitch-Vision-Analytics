import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../lib/api";
import type { InsightsResponse } from "../types/api";

type Coach = InsightsResponse["chase_coach"];

// Below this many training visits the coach has seen a situation too rarely to trust it fully.
const LOW_EVIDENCE_VISITS = 300;
// Win chances this close together mean the approaches are roughly equal.
const CLOSE_CALL_POINTS = 2;

function bucketIndex(value: number, edges: number[]) {
  const index = edges.findIndex((edge) => value <= edge);
  return index === -1 ? edges.length : index;
}

const toWinChance = (qValue: number) => Math.round(((qValue + 1) / 2) * 100);

export function ChaseCoach() {
  const [coach, setCoach] = useState<Coach | null>(null);
  const [balls, setBalls] = useState(60);
  const [runs, setRuns] = useState(85);
  const [wickets, setWickets] = useState(7);

  useEffect(() => {
    api
      .getInsights()
      .then((payload) => setCoach(payload.chase_coach))
      .catch(() => setCoach(null));
  }, []);

  if (!coach) {
    return null;
  }

  const inputError =
    !Number.isInteger(balls) || balls < 1 || balls > 120
      ? "Balls left must be between 1 and 120."
      : !Number.isInteger(runs) || runs < 1
        ? "Runs needed must be at least 1."
        : !Number.isInteger(wickets) || wickets < 1 || wickets > 10
          ? "Wickets in hand must be between 1 and 10."
          : null;
  const requiredRate = inputError ? 0 : (runs * 6) / balls;
  const group = inputError
    ? null
    : {
        balls: coach.balls_buckets[bucketIndex(balls, coach.balls_edges)],
        rate: coach.rate_buckets[bucketIndex(requiredRate, coach.rate_edges)],
        wickets: coach.wickets_buckets[bucketIndex(wickets, coach.wickets_edges)],
      };
  const advice = group
    ? coach.policy.find(
        (row) => row.balls_left === group.balls && row.required_rate === group.rate && row.wickets_in_hand === group.wickets,
      )
    : undefined;
  const chances = advice ? coach.actions.map((action) => toWinChance(advice.q_values[action])).sort((a, b) => b - a) : [];
  const closeCall = chances.length > 1 && chances[0] - chances[1] <= CLOSE_CALL_POINTS;
  const oversLeft = Math.floor(balls / 6);
  const extraBalls = balls % 6;

  return (
    <section className="two-column">
      <article className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Chase Coach</p>
            <h2>How should the batters play this chase?</h2>
          </div>
        </div>
        <p className="muted-copy">
          A coach agent played {coach.episodes.toLocaleString()} simulated chases from any point in the 20 overs and learned by trial and error when to
          defend, rotate the strike or attack. Rotate uses real IPL chase ball outcomes for the current phase (powerplay, middle or death). Defend gives
          up boundaries for safety, and Attack hits more boundaries but loses more wickets.
        </p>
        <div className="form-grid">
          <label>
            Balls left (1-120)
            <input type="number" min={1} max={120} value={balls} onChange={(event) => setBalls(Number(event.target.value))} />
          </label>
          <label>
            Runs needed
            <input type="number" min={1} value={runs} onChange={(event) => setRuns(Number(event.target.value))} />
          </label>
          <label>
            Wickets in hand (1-10)
            <input type="number" min={1} max={10} value={wickets} onChange={(event) => setWickets(Number(event.target.value))} />
          </label>
          <div className="rate-box">
            <span className="muted-copy">Required run rate</span>
            <strong>{inputError ? "-" : requiredRate.toFixed(2)}</strong>
            <span className="muted-copy">{inputError ? "" : `${oversLeft}.${extraBalls} overs left`}</span>
          </div>
        </div>
        {inputError ? <p className="error-copy">{inputError}</p> : null}
        {advice && group ? (
          <div className="sub-panel">
            <span className="muted-copy">Recommended approach</span>
            <p className="coach-action">{closeCall ? `${advice.best_action} (close call)` : advice.best_action}</p>
            <div className="mini-stats">
              {coach.actions.map((action) => (
                <div key={action}>
                  <span>Win chance if they {action.toLowerCase()}</span>
                  <strong>{toWinChance(advice.q_values[action])}%</strong>
                </div>
              ))}
            </div>
            {advice.visits < LOW_EVIDENCE_VISITS ? (
              <p className="error-copy">
                {advice.visits === 0
                  ? "The coach never reached this situation in training, so these numbers are not meaningful."
                  : `This situation is rare (seen ${advice.visits} times in training), so treat these numbers as a rough estimate.`}
              </p>
            ) : null}
            <p className="footnote">
              Each figure is the coach&apos;s learned chance of winning the chase if the batters play that way from here and then keep making the best
              choice. The highest one is the advice{closeCall ? "; here the options are within a couple of points, so any of them is reasonable" : ""}. The
              coach learned on situation groups, so this uses the group {group.balls} balls left, required rate {group.rate}, {group.wickets} wickets.
            </p>
          </div>
        ) : null}
      </article>

      <article className="card chart-card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Learning Curve</p>
            <h3>Coach win rate while training</h3>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={coach.learning_curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#20314e" />
            <XAxis dataKey="episode" stroke="#b9c7dd" />
            <YAxis stroke="#b9c7dd" unit="%" domain={[0, 100]} />
            <Tooltip />
            <Line type="monotone" dataKey="win_rate" name="Win rate" stroke="#1ec8a5" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
        <p className="footnote">
          Win rate on the same {coach.test_chases} test chases (full 20 overs, targets 140-220), checked as training goes on. The coach first learns
          the end of a chase and then works backwards to the start.
        </p>
      </article>
    </section>
  );
}
