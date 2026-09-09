import { useEffect, useMemo, useState } from "react";
import {
  decisionLabel,
  defectLabel,
  fetchInspectionLines,
  fetchInspections,
  InspectionResult,
  statusLabel,
} from "../api";

type ViewMode = "gallery" | "table";

export default function HistoryPage() {
  const [items, setItems] = useState<InspectionResult[]>([]);
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [line, setLine] = useState("DemoEx");
  const [view, setView] = useState<ViewMode>("gallery");
  const [selected, setSelected] = useState<InspectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchInspectionLines()
      .then((ls) => {
        setLines(ls);
        if (ls.includes("DemoEx")) setLine("DemoEx");
        else if (ls.length && !ls.includes(line)) setLine(ls[0]);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchInspections({
      status: status || undefined,
      line: line || undefined,
      limit: 120,
    })
      .then((rows) => {
        setItems(rows);
        setSelected((prev) => {
          if (!prev) return rows.find((r) => r.image_url) ?? rows[0] ?? null;
          return rows.find((r) => r.id === prev.id) ?? rows[0] ?? null;
        });
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [status, line]);

  const counts = useMemo(() => {
    const c = { PASS: 0, FAIL: 0, REVIEW: 0 };
    for (const r of items) c[r.status] = (c[r.status] || 0) + 1;
    return c;
  }, [items]);

  return (
    <div className="page results-page">
      <div className="page-head">
        <div>
          <h1>Rezultati inšpekcij</h1>
          <p className="muted">Galerija slik · original / označbe / toplotna karta · zakaj PASS/FAIL</p>
        </div>
        <div className="filters">
          <select value={line} onChange={(e) => setLine(e.target.value)}>
            <option value="">Vse linije</option>
            {lines.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">Vsi statusi</option>
            <option value="PASS">SPREJETO</option>
            <option value="FAIL">ZAVRNJENO</option>
            <option value="REVIEW">ROČNI PREGLED</option>
          </select>
          <div className="seg">
            <button
              type="button"
              className={view === "gallery" ? "active" : ""}
              onClick={() => setView("gallery")}
            >
              Galerija
            </button>
            <button
              type="button"
              className={view === "table" ? "active" : ""}
              onClick={() => setView("table")}
            >
              Tabela
            </button>
          </div>
        </div>
      </div>

      <div className="stat-grid compact">
        <div className="stat-card">
          <div className="stat-label">Skupaj</div>
          <div className="stat-value">{items.length}</div>
        </div>
        <div className="stat-card ok">
          <div className="stat-label">PASS</div>
          <div className="stat-value">{counts.PASS}</div>
        </div>
        <div className="stat-card bad">
          <div className="stat-label">FAIL</div>
          <div className="stat-value">{counts.FAIL}</div>
        </div>
        <div className="stat-card warn">
          <div className="stat-label">REVIEW</div>
          <div className="stat-value">{counts.REVIEW}</div>
        </div>
      </div>

      {error && <div className="banner error">{error}</div>}
      {loading && <div className="banner">Nalagam rezultate…</div>}

      {!loading && items.length === 0 && (
        <div className="panel empty-state">Ni inšpekcij za izbrane filtre.</div>
      )}

      {view === "gallery" && items.length > 0 && (
        <div className="results-layout">
          <div className="gallery-grid">
            {items.map((r) => (
              <button
                key={r.id}
                type="button"
                className={`gallery-card status-${r.status.toLowerCase()} ${
                  selected?.id === r.id ? "selected" : ""
                }`}
                onClick={() => setSelected(r)}
              >
                <div className="gallery-thumb">
                  {r.image_url ? (
                    <img src={r.image_url} alt={r.part_id} loading="lazy" />
                  ) : (
                    <div className="thumb-empty">Brez slike</div>
                  )}
                  <span className={`tag status-${r.status.toLowerCase()}`}>
                    {statusLabel(r.status)}
                  </span>
                </div>
                <div className="gallery-meta">
                  <strong className="mono">{r.part_id}</strong>
                  <span className="muted small">
                    Q {r.quality_score} · odst. {(r.anomaly_score ?? 0).toFixed(2)}
                  </span>
                </div>
              </button>
            ))}
          </div>

          {selected && (
            <aside className="detail-panel panel">
              <div className={`status-hero status-${selected.status.toLowerCase()}`}>
                <div>
                  <div className="status-label">{statusLabel(selected.status)}</div>
                  <div className="muted small">{decisionLabel(selected.decision)}</div>
                </div>
                <div className="status-score">Q {selected.quality_score}</div>
              </div>

              <p className="reason">{selected.reason || "—"}</p>

              <dl className="kv">
                <div>
                  <dt>Kos</dt>
                  <dd className="mono">{selected.part_id}</dd>
                </div>
                <div>
                  <dt>Linija</dt>
                  <dd>{selected.line}</dd>
                </div>
                <div>
                  <dt>Indeks odstopanja</dt>
                  <dd>{Number(selected.anomaly_score ?? 0).toFixed(3)}</dd>
                </div>
                <div>
                  <dt>Latenca</dt>
                  <dd>{Math.round(selected.latency_ms)} ms</dd>
                </div>
                <div>
                  <dt>Čas</dt>
                  <dd>{new Date(selected.created_at).toLocaleString("sl-SI")}</dd>
                </div>
                <div>
                  <dt>Napake</dt>
                  <dd>
                    {selected.defects?.length
                      ? selected.defects
                          .map((d) => `${defectLabel(d.type)} ${(d.confidence * 100).toFixed(0)}%`)
                          .join(", ")
                      : "—"}
                  </dd>
                </div>
              </dl>

              <div className="result-images three">
                <figure>
                  <img src={selected.image_url || ""} alt="Original" />
                  <figcaption>Original</figcaption>
                </figure>
                <figure>
                  <img
                    src={selected.annotated_url || selected.image_url || ""}
                    alt="Anotacija"
                  />
                  <figcaption>Označene napake</figcaption>
                </figure>
                <figure>
                  {selected.heatmap_url ? (
                    <img src={selected.heatmap_url} alt="Toplotna karta" />
                  ) : (
                    <div className="thumb-empty">Ni toplotne karte</div>
                  )}
                  <figcaption>Toplotna karta odstopanj</figcaption>
                </figure>
              </div>
            </aside>
          )}
        </div>
      )}

      {view === "table" && items.length > 0 && (
        <div className="table-wrap panel">
          <table>
            <thead>
              <tr>
                <th></th>
                <th>Čas</th>
                <th>Kos</th>
                <th>Linija</th>
                <th>Status</th>
                <th>Score</th>
                <th>Napake</th>
                <th>Odstopanje</th>
                <th>Zakaj</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr
                  key={r.id}
                  className={selected?.id === r.id ? "row-selected" : ""}
                  onClick={() => {
                    setSelected(r);
                    setView("gallery");
                  }}
                >
                  <td>
                    {r.image_url && (
                      <img className="table-thumb" src={r.image_url} alt="" />
                    )}
                  </td>
                  <td>{new Date(r.created_at).toLocaleString("sl-SI")}</td>
                  <td className="mono">{r.part_id}</td>
                  <td>{r.line}</td>
                  <td>
                    <span className={`tag status-${r.status.toLowerCase()}`}>
                      {statusLabel(r.status)}
                    </span>
                  </td>
                  <td>{r.quality_score}</td>
                  <td>
                    {r.defects?.length
                      ? r.defects.map((d) => defectLabel(d.type)).join(", ")
                      : "—"}
                  </td>
                  <td>{Number(r.anomaly_score ?? 0).toFixed(3)}</td>
                  <td className="reason-cell">{r.reason || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
