import { FormEvent, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { HeadToHeadResponse, OptionsResponse, PartnershipResponse } from "../types/api";

export function MatchupsPage() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [batter, setBatter] = useState("");
  const [bowler, setBowler] = useState("");
  const [partnerOne, setPartnerOne] = useState("");
  const [partnerTwo, setPartnerTwo] = useState("");
  const [headToHead, setHeadToHead] = useState<HeadToHeadResponse | null>(null);
  const [partnership, setPartnership] = useState<PartnershipResponse | null>(null);

  useEffect(() => {
    api.getOptions().then((payload) => {
      setOptions(payload);
      setBatter(payload.batters[0]?.value ?? "");
      setBowler(payload.bowlers[0]?.value ?? "");
      setPartnerOne(payload.batters[0]?.value ?? "");
      setPartnerTwo(payload.batters[1]?.value ?? payload.batters[0]?.value ?? "");
    });
  }, []);

  const submitHeadToHead = async (event: FormEvent) => {
    event.preventDefault();
    setHeadToHead(await api.headToHead({ batter, bowler }));
  };

  const submitPartnership = async (event: FormEvent) => {
    event.preventDefault();
    setPartnership(await api.partnership({ batter_one: partnerOne, batter_two: partnerTwo }));
  };

  return (
    <section className="two-column">
      <form className="card form-card" onSubmit={submitHeadToHead}>
        <div className="section-head">
          <div>
            <p className="eyebrow">Head-to-Head Matchups</p>
            <h2>Batter vs bowler pressure map</h2>
          </div>
        </div>
        <label>
          Batter
          <select value={batter} onChange={(event) => setBatter(event.target.value)}>
            {options?.batters.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Bowler
          <select value={bowler} onChange={(event) => setBowler(event.target.value)}>
            {options?.bowlers.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <button type="submit">Load matchup</button>
        {headToHead ? (
          <div className="stats-grid">
            <div><span>Runs</span><strong>{headToHead.runs}</strong></div>
            <div><span>Balls</span><strong>{headToHead.balls}</strong></div>
            <div><span>Strike rate</span><strong>{headToHead.strike_rate.toFixed(2)}</strong></div>
            <div><span>Dismissals</span><strong>{headToHead.dismissals}</strong></div>
            <div><span>Boundaries</span><strong>{headToHead.boundaries}</strong></div>
            <div><span>Sixes</span><strong>{headToHead.sixes}</strong></div>
          </div>
        ) : null}
      </form>

      <form className="card form-card" onSubmit={submitPartnership}>
        <div className="section-head">
          <div>
            <p className="eyebrow">Partnership Synergy Matrix</p>
            <h2>Pair chemistry and uplift</h2>
          </div>
        </div>
        <label>
          Batter one
          <select value={partnerOne} onChange={(event) => setPartnerOne(event.target.value)}>
            {options?.batters.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Batter two
          <select value={partnerTwo} onChange={(event) => setPartnerTwo(event.target.value)}>
            {options?.batters.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <button type="submit">Analyze partnership</button>
        {partnership ? (
          <div className="stats-grid">
            <div><span>Runs together</span><strong>{partnership.runs}</strong></div>
            <div><span>Balls</span><strong>{partnership.balls}</strong></div>
            <div><span>Run rate</span><strong>{partnership.run_rate.toFixed(2)}</strong></div>
            <div><span>Shared innings</span><strong>{partnership.shared_innings}</strong></div>
            <div><span>Average stand</span><strong>{partnership.average_partnership.toFixed(2)}</strong></div>
            <div><span>Synergy uplift</span><strong>{partnership.synergy_index.toFixed(2)}</strong></div>
          </div>
        ) : null}
      </form>
    </section>
  );
}

