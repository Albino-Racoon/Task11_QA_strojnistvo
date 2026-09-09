import { useEffect, useState } from "react";
import { defectLabel, fetchDefectStats } from "../api";

export default function DefectsPage() {
  const [items, setItems] = useState<{ type: string; count: number }[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDefectStats()
      .then(setItems)
      .catch((e) => setError(String(e)));
  }, []);

  const max = Math.max(1, ...items.map((i) => i.count));

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Napake</h1>
          <p className="muted">Frekvenca tipov napak v zgodovini</p>
        </div>
      </div>
      {error && <div className="banner error">{error}</div>}
      <div className="panel bars">
        {items.length === 0 && <div className="empty-state">Ni podatkov o napakah.</div>}
        {items.map((item) => (
          <div className="bar-row" key={item.type}>
            <div className="bar-label">{defectLabel(item.type)}</div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(item.count / max) * 100}%` }} />
            </div>
            <div className="bar-count mono">{item.count}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
