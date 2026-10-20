import { NavLink, Outlet } from "react-router-dom";

const tabs = [
  { to: "/", label: "Dashboard" },
  { to: "/predictor", label: "Win Predictor" },
  { to: "/matchups", label: "Matchups" },
  { to: "/styles", label: "Player Styles" },
  { to: "/clutch", label: "Clutch Mode" },
  { to: "/captain", label: "Captain AI" },
  { to: "/impact", label: "Impact Lab" },
  { to: "/proof", label: "Data Proof" },
];

export function AppShell() {
  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>Pitch Vision Analytics</h1>
          <p className="hero-copy">
            Broadcast-grade IPL intelligence with live prediction, clutch analytics, and an evidence-grounded captain assistant.
          </p>
        </div>
        <div className="hero-badge">
          <span>2008-2026</span>
          <strong>Ball-by-ball AI engine</strong>
        </div>
      </header>
      <nav className="tab-nav" aria-label="Primary">
        {tabs.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            end={tab.to === "/"}
            className={({ isActive }) => `tab-link${isActive ? " active" : ""}`}
          >
            {tab.label}
          </NavLink>
        ))}
      </nav>
      <main className="page-grid">
        <Outlet />
      </main>
    </div>
  );
}
