import { useEffect, useRef, useState } from "react";
import {
  decisionLabel,
  defectLabel,
  fetchHealth,
  inspectCamera,
  inspectImage,
  InspectionResult,
  statusLabel,
} from "../api";

export default function InspectPage() {
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [line, setLine] = useState("Linija A");
  const [health, setHealth] = useState<string>("…");
  const [preview, setPreview] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraOn, setCameraOn] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        const d = h.models?.detector?.available;
        const a = h.models?.anomaly?.available;
        setHealth(
          `Detektor ${d ? "pripravljen" : "ni na voljo"} · Odstopanja ${a ? "pripravljena" : "niso na voljo"}`,
        );
      })
      .catch(() => setHealth("API ni dosegljiv"));
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  async function onFile(file: File | null) {
    if (!file) return;
    setError(null);
    setBusy(true);
    setPreview(URL.createObjectURL(file));
    try {
      const res = await inspectImage(file, line);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Napaka pri analizi");
    } finally {
      setBusy(false);
    }
  }

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch {
      setError("Brskalnik ne more odpreti kamere. Poskusite strežniški zajem.");
    }
  }

  function stopCamera() {
    const video = videoRef.current;
    const stream = video?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (video) video.srcObject = null;
    setCameraOn(false);
  }

  async function captureBrowserFrame() {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    setBusy(true);
    setError(null);
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setBusy(false);
        return;
      }
      const file = new File([blob], "camera.jpg", { type: "image/jpeg" });
      setPreview(URL.createObjectURL(blob));
      try {
        const res = await inspectImage(file, line);
        setResult(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Napaka");
      } finally {
        setBusy(false);
      }
    }, "image/jpeg", 0.92);
  }

  async function captureServerCamera() {
    setBusy(true);
    setError(null);
    try {
      const res = await inspectCamera(line);
      setResult(res);
      if (res.annotated_url) setPreview(res.annotated_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Kamera ni dosegljiva");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Pregled kosa</h1>
          <p className="muted">Naloži sliko ali zajemi s kamero · vizualni QC</p>
        </div>
        <div className="pill">{health}</div>
      </div>

      <div className="inspect-grid">
        <section className="panel">
          <div className="panel-title">Vhod</div>
          <label className="field">
            <span>Proizvodna linija</span>
            <select value={line} onChange={(e) => setLine(e.target.value)}>
              <option>Linija A</option>
              <option>Linija B</option>
              <option>Linija C</option>
            </select>
          </label>

          <label className="dropzone">
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => onFile(e.target.files?.[0] || null)}
            />
            <strong>Izberi sliko</strong>
            <span>JPG / PNG · kovinska površina</span>
          </label>

          <div className="btn-row">
            {!cameraOn ? (
              <button className="btn" onClick={startCamera} disabled={busy}>
                Živa kamera (brskalnik)
              </button>
            ) : (
              <>
                <button className="btn primary" onClick={captureBrowserFrame} disabled={busy}>
                  Zajemi okvir
                </button>
                <button className="btn" onClick={stopCamera}>
                  Ustavi
                </button>
              </>
            )}
            <button className="btn" onClick={captureServerCamera} disabled={busy}>
              USB kamera (strežnik)
            </button>
          </div>

          <div className="preview-wrap">
            {cameraOn && <video ref={videoRef} className="preview" muted playsInline />}
            {!cameraOn && preview && <img src={preview} alt="Vhod" className="preview" />}
            {!cameraOn && !preview && <div className="preview empty">Ni slike</div>}
          </div>
          {busy && <div className="banner">Analiziram…</div>}
          {error && <div className="banner error">{error}</div>}
        </section>

        <section className="panel">
          <div className="panel-title">Rezultat</div>
          {!result ? (
            <div className="empty-state">Rezultat se prikaže po analizi.</div>
          ) : (
            <>
              <div className={`status-hero status-${result.status.toLowerCase()}`}>
                <div className="status-label">{statusLabel(result.status)}</div>
                <div className="status-score">{result.quality_score}/100</div>
              </div>
              <dl className="kv">
                <div>
                  <dt>Št. kosa</dt>
                  <dd>{result.part_id}</dd>
                </div>
                <div>
                  <dt>Odločitev</dt>
                  <dd>{decisionLabel(result.decision)}</dd>
                </div>
                <div>
                  <dt>Indeks odstopanja</dt>
                  <dd>{((result.anomaly?.score ?? 0) * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Latenca</dt>
                  <dd>{result.latency_ms.toFixed(0)} ms</dd>
                </div>
              </dl>
              <p className="reason">{result.reason}</p>
              {result.demo_fallback && (
                <div className="banner warn">
                  Demo način — modeli še niso v celoti naloženi.
                </div>
              )}
              {result.defects.length > 0 && (
                <div className="defects">
                  <div className="panel-title">Zaznane napake</div>
                  {result.defects.map((d, i) => (
                    <div className="defect-row" key={i}>
                      <span>{defectLabel(d.type)}</span>
                      <span>{(d.confidence * 100).toFixed(1)}%</span>
                      <span>{d.severity}</span>
                      <span>{(d.affected_area * 100).toFixed(2)}% površine</span>
                    </div>
                  ))}
                </div>
              )}
              <div className="result-images">
                {result.annotated_url && (
                  <figure>
                    <img src={result.annotated_url} alt="Označeno" />
                    <figcaption>Označene napake</figcaption>
                  </figure>
                )}
                {result.anomaly?.heatmap_url && (
                  <figure>
                    <img src={result.anomaly.heatmap_url} alt="Toplotna karta" />
                    <figcaption>Toplotna karta odstopanj</figcaption>
                  </figure>
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
