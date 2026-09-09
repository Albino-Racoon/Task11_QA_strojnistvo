import { useEffect, useState } from "react";
import { defectLabel, fetchTodayStats, TodayStats } from "../api";

export default function AnalyticsPage() {
  const [stats, setStats] = useState<TodayStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTodayStats()
      .then(setStats)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Analitika proizvodnje</h1>
          <p className="muted">
            {stats?.scope === "today" ? "Danes" : "Zadnji zapisi (danes še ni podatkov)"}
          </p>
        </div>
      </div>
      {error && <div className="banner error">{error}</div>}
      {stats && (
        <>
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">Proizvedeno</div>
              <div className="stat-value">{stats.produced.toLocaleString("sl-SI")}</div>
            </div>
            <div className="stat-card ok">
              <div className="stat-label">Sprejeto</div>
              <div className="stat-value">{stats.passed.toLocaleString("sl-SI")}</div>
            </div>
            <div className="stat-card bad">
              <div className="stat-label">Zavrnjeno</div>
              <div className="stat-value">{stats.rejected.toLocaleString("sl-SI")}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Stopnja napak</div>
              <div className="stat-value">{stats.defect_rate.toFixed(2)} %</div>
            </div>
          </div>

          <div className="two-col">
            <section className="panel">
              <div className="panel-title">Najpogostejše napake</div>
              <table>
                <tbody>
                  {stats.top_defects.map((d) => (
                    <tr key={d.type}>
                      <td>{defectLabel(d.type)}</td>
                      <td className="mono">{d.count}</td>
                    </tr>
                  ))}
                  {stats.top_defects.length === 0 && (
                    <tr>
                      <td colSpan={2}>Ni napak</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </section>
            <section className="panel">
              <div className="panel-title">Stopnja napak po linijah</div>
              <table>
                <thead>
                  <tr>
                    <th>Linija</th>
                    <th>Proizvedeno</th>
                    <th>Napak</th>
                    <th>Stopnja</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.lines.map((l) => (
                    <tr key={l.line} className={l.warning ? "warn-row" : ""}>
                      <td>
                        {l.line} {l.warning ? "⚠" : ""}
                      </td>
                      <td className="mono">{l.produced}</td>
                      <td className="mono">{l.failed}</td>
                      <td className="mono">{l.defect_rate.toFixed(1)} %</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">Povp. latenca: {stats.avg_latency_ms} ms</p>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
