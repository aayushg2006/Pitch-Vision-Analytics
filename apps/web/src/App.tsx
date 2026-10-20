import { BrowserRouter, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { CaptainAIPage } from "./pages/CaptainAIPage";
import { ClutchModePage } from "./pages/ClutchModePage";
import { DashboardPage } from "./pages/DashboardPage";
import { DataProofPage } from "./pages/DataProofPage";
import { ImpactLabPage } from "./pages/ImpactLabPage";
import { MatchupsPage } from "./pages/MatchupsPage";
import { PlayerStylesPage } from "./pages/PlayerStylesPage";
import { WinPredictorPage } from "./pages/WinPredictorPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="predictor" element={<WinPredictorPage />} />
          <Route path="matchups" element={<MatchupsPage />} />
          <Route path="styles" element={<PlayerStylesPage />} />
          <Route path="clutch" element={<ClutchModePage />} />
          <Route path="captain" element={<CaptainAIPage />} />
          <Route path="impact" element={<ImpactLabPage />} />
          <Route path="proof" element={<DataProofPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

