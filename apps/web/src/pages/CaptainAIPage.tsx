import { FormEvent, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { CaptainAdviceRequest, CaptainResponse, OptionItem, OptionsResponse, TeamPlayerPool } from "../types/api";

const BATTER_LIMIT = 2;
const BOWLER_LIMIT = 9;

function getTeamPlayers(pools: TeamPlayerPool[] | undefined, team: string): OptionItem[] {
  return pools?.find((pool) => pool.team === team)?.players ?? [];
}

function pickValidSelections(selected: string[], available: OptionItem[], limit: number): string[] {
  const validSelected = selected.filter((value) => available.some((option) => option.value === value)).slice(0, limit);
  if (validSelected.length > 0) {
    return validSelected;
  }
  return available.slice(0, limit).map((option) => option.value);
}

function toggleSelection(selected: string[], value: string, limit: number): string[] {
  if (selected.includes(value)) {
    return selected.filter((item) => item !== value);
  }
  if (selected.length >= limit) {
    return selected;
  }
  return [...selected, value];
}

export function CaptainAIPage() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [result, setResult] = useState<CaptainResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useAllBatters, setUseAllBatters] = useState(false);
  const [useAllBowlers, setUseAllBowlers] = useState(false);
  const [form, setForm] = useState<CaptainAdviceRequest>({
    innings: 2,
    current_score: 132,
    wickets_lost: 4,
    overs_completed: 15,
    balls_into_over: 0,
    target_score: 185,
    batting_team: "",
    bowling_team: "",
    venue: "",
    batters: [],
    available_bowlers: [],
    remaining_overs_by_bowler: {},
  });

  useEffect(() => {
    api.getOptions().then((payload) => {
      const battingTeam = payload.teams[0]?.value ?? "";
      const bowlingTeam = payload.teams[1]?.value ?? payload.teams[0]?.value ?? "";
      const teamBatters = getTeamPlayers(payload.captain_batters_by_team, battingTeam);
      const teamBowlers = getTeamPlayers(payload.captain_bowlers_by_team, bowlingTeam);
      setOptions(payload);
      setForm((current) => ({
        ...current,
        batting_team: battingTeam,
        bowling_team: bowlingTeam,
        venue: payload.venues[0]?.value ?? "",
        batters: pickValidSelections([], teamBatters.length ? teamBatters : payload.batters, BATTER_LIMIT),
        available_bowlers: pickValidSelections([], teamBowlers.length ? teamBowlers : payload.bowlers, BOWLER_LIMIT),
        remaining_overs_by_bowler: Object.fromEntries(
          pickValidSelections([], teamBowlers.length ? teamBowlers : payload.bowlers, BOWLER_LIMIT).map((bowler) => [bowler, 0]),
        ),
      }));
    });
  }, []);

  const visibleBatters = options
    ? (useAllBatters ? options.batters : getTeamPlayers(options.captain_batters_by_team, form.batting_team))
    : [];
  const visibleBowlers = options
    ? (useAllBowlers ? options.bowlers : getTeamPlayers(options.captain_bowlers_by_team, form.bowling_team))
    : [];

  const syncBatters = (nextTeam: string, nextUseAllBatters: boolean) => {
    if (!options) {
      return;
    }
    const nextPool = nextUseAllBatters ? options.batters : getTeamPlayers(options.captain_batters_by_team, nextTeam);
    setForm((current) => ({
      ...current,
      batting_team: nextTeam,
      batters: pickValidSelections(current.batters, nextPool.length ? nextPool : options.batters, BATTER_LIMIT),
    }));
  };

  const syncBowlers = (nextTeam: string, nextUseAllBowlers: boolean) => {
    if (!options) {
      return;
    }
    const nextPool = nextUseAllBowlers ? options.bowlers : getTeamPlayers(options.captain_bowlers_by_team, nextTeam);
    setForm((current) => ({
      ...current,
      bowling_team: nextTeam,
      ...(() => {
        const nextBowlers = pickValidSelections(current.available_bowlers, nextPool.length ? nextPool : options.bowlers, BOWLER_LIMIT);
        return {
          available_bowlers: nextBowlers,
          remaining_overs_by_bowler: Object.fromEntries(
            nextBowlers.map((bowler) => [bowler, current.remaining_overs_by_bowler[bowler] ?? 0]),
          ),
        };
      })(),
    }));
  };

  const updateBowlerSelection = (bowler: string) => {
    setForm((current) => {
      const nextBowlers = toggleSelection(current.available_bowlers, bowler, BOWLER_LIMIT);
      return {
        ...current,
        available_bowlers: nextBowlers,
        remaining_overs_by_bowler: Object.fromEntries(
          nextBowlers.map((name) => [name, name === bowler && !current.available_bowlers.includes(bowler) ? 0 : current.remaining_overs_by_bowler[name] ?? 0]),
        ),
      };
    });
  };

  const updateRemainingOvers = (bowler: string, value: string) => {
    const parsed = Number(value);
    const sanitized = Number.isFinite(parsed) ? Math.max(0, Math.min(4, Math.trunc(parsed))) : 0;
    setForm((current) => ({
      ...current,
      remaining_overs_by_bowler: {
        ...current.remaining_overs_by_bowler,
        [bowler]: sanitized,
      },
    }));
  };

  const totalRemainingOvers = form.available_bowlers.reduce(
    (sum, bowler) => sum + (form.remaining_overs_by_bowler[bowler] ?? 0),
    0,
  );
  const nextOverHorizon = Math.min(5, Math.max(0, 20 - form.overs_completed - (form.balls_into_over > 0 ? 1 : 0)));
  const cannotPlanAnyOver = totalRemainingOvers <= 0;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (cannotPlanAnyOver) {
      setError("Add at least one legal over remaining across the selected bowlers before asking Captain AI.");
      setResult(null);
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.captainAdvice(form));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Captain AI could not generate advice right now.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="two-column">
      <form className="card form-card" onSubmit={submit}>
        <div className="section-head">
          <div>
            <p className="eyebrow">Agentic Captain Simulator</p>
            <h2>Evidence-grounded tactical advice</h2>
          </div>
        </div>
        <div className="form-grid">
          <label>
            Innings context
            <select
              value={form.innings}
              onChange={(event) => {
                const innings = Number(event.target.value) as 1 | 2;
                setForm((current) => ({
                  ...current,
                  innings,
                  target_score: innings === 1 ? undefined : current.target_score ?? 185,
                }));
              }}
            >
              <option value={1}>1st innings: your team is bowling first</option>
              <option value={2}>2nd innings: your team is defending a target</option>
            </select>
          </label>
          <label>
            Batting team
            <select value={form.batting_team} onChange={(event) => syncBatters(event.target.value, useAllBatters)}>
              {options?.teams.map((team) => (
                <option key={team.value} value={team.value}>
                  {team.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Bowling team
            <select value={form.bowling_team} onChange={(event) => syncBowlers(event.target.value, useAllBowlers)}>
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
            Current score
            <input type="number" value={form.current_score} onChange={(event) => setForm({ ...form, current_score: Number(event.target.value) })} />
          </label>
          <label>
            Wickets lost
            <input type="number" value={form.wickets_lost} onChange={(event) => setForm({ ...form, wickets_lost: Number(event.target.value) })} />
          </label>
          <label>
            Overs completed
            <input type="number" value={form.overs_completed} onChange={(event) => setForm({ ...form, overs_completed: Number(event.target.value) })} />
          </label>
          <label>
            Balls into over
            <input type="number" value={form.balls_into_over} onChange={(event) => setForm({ ...form, balls_into_over: Number(event.target.value) })} />
          </label>
          {form.innings === 2 ? (
            <label>
              Target
              <input
                type="number"
                value={form.target_score ?? ""}
                onChange={(event) =>
                  setForm({
                    ...form,
                    target_score: event.target.value === "" ? undefined : Number(event.target.value),
                  })
                }
              />
            </label>
          ) : (
            <div className="context-note">
              <strong>First-innings mode</strong>
              <span>Captain AI will optimize for phase control, wicket-taking pressure, and run suppression without needing a target.</span>
            </div>
          )}
          <label className="full-width">
            <span className="field-header">
              <span>Batters at crease</span>
              <button
                type="button"
                className={`mode-toggle${useAllBatters ? " active" : ""}`}
                onClick={() => {
                  const nextUseAllBatters = !useAllBatters;
                  setUseAllBatters(nextUseAllBatters);
                  syncBatters(form.batting_team, nextUseAllBatters);
                }}
              >
                {useAllBatters ? "Showing all players" : "Team players only"}
              </button>
            </span>
            <span className="selection-meta">
              Select up to {BATTER_LIMIT} batters. Chosen: {form.batters.length}/{BATTER_LIMIT}
            </span>
            <div className="checkbox-list" role="group" aria-label="Batters at crease">
              {(visibleBatters.length ? visibleBatters : options?.batters ?? []).map((item) => {
                const checked = form.batters.includes(item.value);
                const disabled = !checked && form.batters.length >= BATTER_LIMIT;
                return (
                  <label key={item.value} className={`checkbox-item${checked ? " checked" : ""}${disabled ? " disabled" : ""}`}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={disabled}
                      onChange={() =>
                        setForm((current) => ({
                          ...current,
                          batters: toggleSelection(current.batters, item.value, BATTER_LIMIT),
                        }))
                      }
                    />
                    <span>{item.label}</span>
                  </label>
                );
              })}
            </div>
            <small className="helper-copy">
              {useAllBatters
                ? "Fallback mode is showing the global batter list."
                : `Showing players who have batted for ${form.batting_team}.`}
            </small>
          </label>
          <label className="full-width">
            <span className="field-header">
              <span>Available bowlers</span>
              <button
                type="button"
                className={`mode-toggle${useAllBowlers ? " active" : ""}`}
                onClick={() => {
                  const nextUseAllBowlers = !useAllBowlers;
                  setUseAllBowlers(nextUseAllBowlers);
                  syncBowlers(form.bowling_team, nextUseAllBowlers);
                }}
              >
                {useAllBowlers ? "Showing all players" : "Team players only"}
              </button>
            </span>
            <span className="selection-meta">
              Select up to {BOWLER_LIMIT} bowlers. Chosen: {form.available_bowlers.length}/{BOWLER_LIMIT}
            </span>
            <div className="checkbox-list" role="group" aria-label="Available bowlers">
              {(visibleBowlers.length ? visibleBowlers : options?.bowlers ?? []).map((item) => {
                const checked = form.available_bowlers.includes(item.value);
                const disabled = !checked && form.available_bowlers.length >= BOWLER_LIMIT;
                return (
                  <label key={item.value} className={`checkbox-item${checked ? " checked" : ""}${disabled ? " disabled" : ""}`}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={disabled}
                      onChange={() => updateBowlerSelection(item.value)}
                    />
                    <span>{item.label}</span>
                  </label>
                );
              })}
            </div>
            <small className="helper-copy">
              {useAllBowlers
                ? "Fallback mode is showing the global bowler list."
                : `Showing players who have bowled for ${form.bowling_team}. Captain AI will only recommend phase-qualified options in this innings context.`}
            </small>
          </label>
          <div className="full-width bowler-overs-card">
            <span className="field-header">
              <span>Remaining overs for selected bowlers</span>
              <span className="selection-meta">
                Legal overs available: {totalRemainingOvers} | next-plan horizon: {nextOverHorizon}
              </span>
            </span>
            {form.available_bowlers.length > 0 ? (
              <div className="remaining-overs-grid">
                {form.available_bowlers.map((bowler) => (
                  <label key={bowler} className="remaining-over-row">
                    <span>{bowler}</span>
                    <input
                      type="number"
                      min={0}
                      max={4}
                      step={1}
                      value={form.remaining_overs_by_bowler[bowler] ?? 0}
                      onChange={(event) => updateRemainingOvers(bowler, event.target.value)}
                    />
                  </label>
                ))}
              </div>
            ) : (
              <p className="muted-copy">Select bowlers first, then enter how many legal overs each one has left.</p>
            )}
            <small className="helper-copy">
              Captain AI treats these as hard caps, never assigns a bowler more than this value, and never gives consecutive overs to the same bowler.
            </small>
            {totalRemainingOvers < nextOverHorizon && nextOverHorizon > 0 ? (
              <small className="error-copy">
                The selected bowlers do not have enough legal overs to cover all of the next {nextOverHorizon} overs, so some overs may be left unassigned.
              </small>
            ) : null}
          </div>
        </div>
        <button type="submit" disabled={isLoading || cannotPlanAnyOver}>
          {isLoading ? "Generating tactical advice..." : cannotPlanAnyOver ? "Add legal overs to continue" : "Ask Captain AI"}
        </button>
      </form>

      <section className="card result-card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Tactical Output</p>
            <h2>{isLoading ? "Captain AI is reading the phase context" : result ? "Five-over plan ready" : "Awaiting game state"}</h2>
          </div>
        </div>
        {isLoading ? (
          <div className="loading-state">
            <div className="loading-orb" aria-hidden="true" />
            <strong>Building a phase-wise bowling plan</strong>
            <p className="muted-copy">
              Captain AI is checking the current phase, matchup history, legal remaining overs, and non-consecutive bowling options for this innings situation.
            </p>
            <div className="loading-bars" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
          </div>
        ) : error ? (
          <p className="error-copy">{error}</p>
        ) : result ? (
          <>
            <p className="result-highlight">{result.answer}</p>
            <div className="plan-grid">
              {result.over_plan.map((item) => (
                <article key={`${item.over_number}-${item.phase}`} className="plan-card">
                  <span className="plan-over">Over {item.over_number}</span>
                  <strong>{item.recommended_bowler ?? "No qualified option"}</strong>
                  <span className="plan-phase">{item.phase}</span>
                  <p>{item.reason}</p>
                  <small className="helper-copy">
                    Confidence {(item.confidence * 100).toFixed(0)}%
                    {item.debut_exception_used ? " | debut exception" : ""}
                  </small>
                </article>
              ))}
            </div>
            <div className="mini-stats">
              <div>
                <span>Confidence</span>
                <strong>{(result.confidence * 100).toFixed(0)}%</strong>
              </div>
              <div>
                <span>LLM fallback</span>
                <strong>{result.fallback_used ? "Used" : "Not needed"}</strong>
              </div>
            </div>
            <ul className="evidence-list">
              {result.evidence.map((item) => (
                <li key={item.title}>
                  <strong>{item.title}</strong>
                  <span>{item.stat}</span>
                  <p>{item.detail}</p>
                </li>
              ))}
            </ul>
            <p className="footnote">Contexts: {result.context_ids.join(", ") || "retrieval evidence not required"}</p>
          </>
        ) : (
          <p className="muted-copy">Captain AI now supports both first-innings bowling plans and target-defense scenarios, and it maps the next five overs from your selected bowling options instead of stopping at one move.</p>
        )}
      </section>
    </section>
  );
}
