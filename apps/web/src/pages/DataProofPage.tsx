import { useEffect, useState } from "react";

import { ModelProof } from "../components/ModelProof";
import { api } from "../lib/api";
import type { DataProofResponse } from "../types/api";

export function DataProofPage() {
  const [proof, setProof] = useState<DataProofResponse | null>(null);

  useEffect(() => {
    api.getDataProof().then(setProof);
  }, []);

  if (!proof) {
    return <section className="card">Loading data proof artifacts…</section>;
  }

  return (
    <>
      <section className="card">
        <div className="section-head">
          <div>
            <p className="eyebrow">Hackathon Submission Proof</p>
            <h2>Cleaning lineage and audit summary</h2>
          </div>
        </div>
        <div className="stats-grid">
          <div><span>Raw rows</span><strong>{proof.rows_in_raw}</strong></div>
          <div><span>Matches</span><strong>{proof.matches_in_dataset}</strong></div>
          <div><span>Source file</span><strong>{proof.source_file}</strong></div>
          <div><span>Cleaning script</span><strong>{proof.cleaning_script}</strong></div>
        </div>
        <section className="proof-grid">
          <article>
            <h3>Seasons</h3>
            <p>{proof.seasons.join(", ")}</p>
          </article>
          <article>
            <h3>Derived fields</h3>
            <p>{proof.derived_fields.join(", ")}</p>
          </article>
          <article>
            <h3>Alias mappings</h3>
            <pre>{JSON.stringify(proof.mappings, null, 2)}</pre>
          </article>
        </section>
      </section>
      <ModelProof />
    </>
  );
}

