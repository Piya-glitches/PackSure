import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";

interface ScanSummary {
  id: string;
  created_at: string;
  product_label: string;
  overall_status: string;
  compliance_score: number;
  calibration_source: string | null;
}

const PILL: Record<string, string> = {
  COMPLIANT: "pill-pass",
  NON_COMPLIANT: "pill-fail",
  NEEDS_REVIEW: "pill-warn",
};

const FILTERS = [
  { value: "", label: "All" },
  { value: "COMPLIANT", label: "Compliant" },
  { value: "NON_COMPLIANT", label: "Non-Compliant" },
  { value: "NEEDS_REVIEW", label: "Needs Review" },
];

export default function HistoryPage() {
  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .get("/scans", { params: filter ? { status: filter } : {} })
      .then((r) => setScans(r.data))
      .finally(() => setLoading(false));
  }, [filter]);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-ink mb-1">Scan history</h1>
          <p className="text-sm text-muted">Repository of every scanned product and its inspection outcome.</p>
        </div>
        <div className="flex gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              className={`px-3 py-1.5 text-xs rounded-md ${filter === f.value ? "bg-brand-100 text-brand-700 font-medium" : "text-muted hover:text-ink"}`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <p className="text-sm text-muted">Loading…</p>
      ) : scans.length === 0 ? (
        <div className="card p-10 text-center">
          <p className="text-sm text-muted mb-4">No scans yet.</p>
          <Link to="/scan" className="btn-primary inline-block">Scan a label</Link>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left">
                <th className="px-4 py-3 text-xs text-muted font-medium uppercase tracking-wide">Product</th>
                <th className="px-4 py-3 text-xs text-muted font-medium uppercase tracking-wide">Scanned</th>
                <th className="px-4 py-3 text-xs text-muted font-medium uppercase tracking-wide">Status</th>
                <th className="px-4 py-3 text-xs text-muted font-medium uppercase tracking-wide">Score</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {scans.map((s) => (
                <tr key={s.id} className="border-b border-line last:border-0 hover:bg-canvas">
                  <td className="px-4 py-3 text-ink">{s.product_label}</td>
                  <td className="px-4 py-3 mono text-xs text-muted">{new Date(s.created_at).toLocaleString("en-IN")}</td>
                  <td className="px-4 py-3"><span className={`pill ${PILL[s.overall_status]}`}>{s.overall_status.replace("_", " ")}</span></td>
                  <td className="px-4 py-3 mono text-ink">{s.compliance_score}</td>
                  <td className="px-4 py-3 text-right"><Link to={`/scans/${s.id}`} className="text-brand-700 text-xs font-medium hover:underline">View →</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
