import { NavLink, Route, Routes } from "react-router-dom";
import InspectPage from "./pages/InspectPage";
import HistoryPage from "./pages/HistoryPage";
import DefectsPage from "./pages/DefectsPage";
import AnalyticsPage from "./pages/AnalyticsPage";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" />
          <div>
            <div className="brand-title">AI Visual Quality Inspector</div>
            <div className="brand-sub">Kovinski deli · vizualni QC</div>
          </div>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Pregled
          </NavLink>
          <NavLink to="/zgodovina">Rezultati</NavLink>
          <NavLink to="/napake">Napake</NavLink>
          <NavLink to="/analitika">Analitika</NavLink>
        </nav>
      </header>
      <main className="main">
        <Routes>
          <Route path="/" element={<InspectPage />} />
          <Route path="/zgodovina" element={<HistoryPage />} />
          <Route path="/napake" element={<DefectsPage />} />
          <Route path="/analitika" element={<AnalyticsPage />} />
        </Routes>
      </main>
    </div>
  );
}
