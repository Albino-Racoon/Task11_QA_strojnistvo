export type Defect = {
  type: string;
  confidence: number;
  bbox: number[];
  severity: string;
  affected_area: number;
};

export type InspectionResult = {
  id: number;
  part_id: string;
  line: string;
  status: "PASS" | "FAIL" | "REVIEW";
  decision: string;
  quality_score: number;
  reason?: string;
  defects: Defect[];
  anomaly?: {
    score: number;
    heatmap_url?: string | null;
    is_anomaly?: boolean;
  };
  anomaly_score?: number;
  latency_ms: number;
  image_url?: string | null;
  annotated_url?: string | null;
  heatmap_url?: string | null;
  created_at: string;
  demo_fallback?: boolean;
};

export type TodayStats = {
  scope: string;
  produced: number;
  passed: number;
  rejected: number;
  review: number;
  defect_rate: number;
  top_defects: { type: string; count: number }[];
  lines: { line: string; produced: number; failed: number; defect_rate: number; warning: boolean }[];
  avg_latency_ms: number;
};

const API = "/api";

export async function inspectImage(file: File, line = "Linija A"): Promise<InspectionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("line", line);
  const res = await fetch(`${API}/inspect`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function inspectCamera(line = "Linija A"): Promise<InspectionResult> {
  const res = await fetch(`${API}/inspect/camera?line=${encodeURIComponent(line)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchInspections(opts?: {
  status?: string;
  line?: string;
  limit?: number;
}): Promise<InspectionResult[]> {
  const params = new URLSearchParams();
  if (opts?.status) params.set("status", opts.status);
  if (opts?.line) params.set("line", opts.line);
  if (opts?.limit) params.set("limit", String(opts.limit));
  const q = params.toString() ? `?${params}` : "";
  const res = await fetch(`${API}/inspections${q}`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.items;
}

export async function fetchInspectionLines(): Promise<string[]> {
  const res = await fetch(`${API}/inspections/lines`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.items;
}

export async function fetchTodayStats(): Promise<TodayStats> {
  const res = await fetch(`${API}/stats/today`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchDefectStats(): Promise<{ type: string; count: number }[]> {
  const res = await fetch(`${API}/stats/defects`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.items;
}

export async function fetchHealth() {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function statusLabel(status: string): string {
  switch (status) {
    case "PASS":
      return "SPREJETO";
    case "FAIL":
      return "ZAVRNJENO";
    case "REVIEW":
      return "ROČNI PREGLED";
    default:
      return status;
  }
}

export function decisionLabel(decision: string): string {
  switch (decision) {
    case "ACCEPT":
      return "Sprejmi";
    case "REJECT":
      return "Zavrni";
    case "MANUAL_REVIEW":
      return "Zahtevan ročni pregled";
    default:
      return decision;
  }
}

export function defectLabel(type: string): string {
  const map: Record<string, string> = {
    scratch: "Praska",
    inclusion: "Vključek",
    oil_spot: "Oljni madež",
    water_spot: "Vodni madež",
    welding_line: "Varilni šiv",
    punching_hole: "Prebojna luknja",
    rolled_pit: "Valjana jama",
    crease: "Guba",
    waist_folding: "Pasna guba",
    crescent_gap: "Polmesečna reža",
    silk_spot: "Svilen madež",
    other_anomaly: "Druga anomalija",
    crazing: "Mrežaste razpoke",
    patches: "Lise",
    pitted_surface: "Jamičasta površina",
    misrun: "Nepopolno litje",
    dent: "Udretek",
    parting_line_crack: "Razpoka ločilne linije",
    stamp_collapse: "Ugrez žiga",
    pockmarks: "Jamičaste luknje",
    mould_scuffing: "Poškodba kalupa",
    cut_marks: "Rezi / nožne sledi",
  };
  return map[type] || type;
}
