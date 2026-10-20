import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "../lib/api";
import type { BatterPoint, InsightsResponse } from "../types/api";

const STYLE_COLORS: Record<string, string> = {
  "Power-hitter": "#f6b94d",
  Anchor: "#1ec8a5",
  Accumulator: "#5cbcff",
  Grinder: "#ff7a7a",
};

const FEATURE_LABELS: Record<string, string> = {
  strike_rate: "Strike rate",
  average: "Average",
  boundary_pct: "Boundary %",
  dot_pct: "Dot ball %",
};

// Recharts copies point fields onto the SVG symbol, so the cluster name can't be called `style`.
type ChartPoint = Omit<BatterPoint, "style"> & { group: string };

function PlayerTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ChartPoint }> }) {
  if (!active || !payload?.length) {
    return null;
  }
  const point = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{point.player}</strong>
      <div className="tooltip-row">
        <small>{point.group}</small>
        <span>
          SR {point.strike_rate} · Avg {point.average}
        </span>
      </div>
    </div>
  );
}

export function PlayerStylesPage() {
  const [insights, setInsights] = useState<InsightsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getInsights().then(setInsights).catch((reason) => setError(String(reason)));
  }, []);

  if (error) {
    return <section className="card error-card">Unable to load player styles: {error}</section>;
  }

  if (!insights) {
    return <section className="card">Loading player styles...</section>;
  }

  const styles = insights.batter_styles;
  const [pc1, pc2] = styles.explained_variance;

  return (
    <>
      <section className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Batter Styles</p>
            <h2>Every IPL batter grouped into {styles.k} playing styles</h2>
          </div>
          <span className="scope-pill">{styles.points.length} batters with 300+ balls</span>
        </div>
        <p className="muted-copy">
          Batters are grouped by strike rate, average, boundary % and dot-ball %. Similar batters land in the same style.
        </p>
        <div className="style-grid">
          {styles.clusters.map((cluster) => (
            <article key={cluster.cluster} className="style-card" style={{ borderColor: STYLE_COLORS[cluster.style] }}>
              <strong style={{ color: STYLE_COLORS[cluster.style] }}>{cluster.style}</strong>
              <small>{cluster.size} batters</small>
              {Object.entries(cluster.centroid).map(([feature, value]) => (
                <small key={feature}>
                  {FEATURE_LABELS[feature] ?? feature}: {value.toFixed(1)}
                </small>
              ))}
              <span>{cluster.top_players.join(", ")}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="two-column">
        <article className="card chart-card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Style Map</p>
              <h3>All batters on one 2-D map</h3>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#20314e" />
              <XAxis type="number" dataKey="x" name="Component 1" stroke="#b9c7dd" tick={false} />
              <YAxis type="number" dataKey="y" name="Component 2" stroke="#b9c7dd" tick={false} />
              <Tooltip content={<PlayerTooltip />} />
              <Legend />
              {styles.clusters.map((cluster) => (
                <Scatter
                  key={cluster.cluster}
                  name={cluster.style}
                  data={styles.points
                    .filter((point) => point.style === cluster.style)
                    .map(({ style, ...point }): ChartPoint => ({ ...point, group: style }))}
                  fill={STYLE_COLORS[cluster.style]}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
          <p className="footnote">
            Four stats are compressed into two axes that keep {(pc1 + pc2).toFixed(1)}% of the information ({pc1}% + {pc2}%).
          </p>
        </article>

        <article className="card chart-card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Why {styles.k} styles?</p>
              <h3>Elbow check</h3>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={styles.elbow}>
              <CartesianGrid strokeDasharray="3 3" stroke="#20314e" />
              <XAxis dataKey="k" stroke="#b9c7dd" />
              <YAxis stroke="#b9c7dd" />
              <Tooltip />
              <Line type="monotone" dataKey="inertia" name="Within-group spread" stroke="#f6b94d" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <p className="footnote">The spread drops sharply until about {styles.k} groups, then flattens out.</p>
        </article>
      </section>
    </>
  );
}
