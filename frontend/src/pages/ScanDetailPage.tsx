import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import { FIELD_LABELS } from "../types";

const STATUS_META: Record<string, { label: string; color: string }> = {
  COMPLIANT: { label: "Compliant", color: "text-brand-700" },
  NON_COMPLIANT: { label: "Non-Compliant", color: "text-red-600" },
  NEEDS_REVIEW: { label: "Needs Manual Review", color: "text-amber" },
};

const FIELD_PILL: Record<string, string> = {
  PASS: "pill-pass",
  FAIL: "pill-fail",
  WARN: "pill-warn",
  NOT_FOUND: "pill-neutral",
};

export default function ScanDetailPage() {
  const { id } = useParams();
  const [scan, setScan] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get(`/scans/${id}`).then((r) => setScan(r.data)).finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="mx-auto max-w-4xl px-6 py-10 text-sm text-muted">Loading…</div>;
  if (!scan) return <div className="mx-auto max-w-4xl px-6 py-10 text-sm text-red-600">Scan not found.</div>;

  const meta = STATUS_META[scan.overall_status];

  return (
    <div className="mx-auto max-w-4xl px-6 py-10 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-ink mb-1">{scan.product_label}</h1>
          <p className="mono text-xs text-muted">{new Date(scan.created_at).toLocaleString("en-IN")} · {scan.id}</p>
        </div>
        <a href={`${api.defaults.baseURL}/scans/${scan.id}/report`} target="_blank" rel="noreferrer" className="btn-primary">
          Download PDF
        </a>
      </div>

      <div className="card p-6 flex items-center justify-between">
        <h2 className={`text-xl font-semibold ${meta.color}`}>{meta.label}</h2>
        <span className="mono text-3xl text-ink">{scan.compliance_score}<span className="text-base text-muted">/100</span></span>
      </div>

      {scan.image_data_url && <img src={scan.image_data_url} alt={scan.product_label} className="rounded-lg border border-line max-w-sm" />}

      <div className="card p-5">
        <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">Mandatory field checklist</p>
        <div className="space-y-3">
          {scan.fields.map((f: any) => (
            <div key={f.field_key} className="border-b border-line pb-3 last:border-0 last:pb-0">
              <div className="flex items-center justify-between mb-1">
                <p className="text-sm text-ink font-medium">{FIELD_LABELS[f.field_key] ?? f.field_key}</p>
                <span className={`pill ${FIELD_PILL[f.status]}`}>{f.status.replace("_", " ")}</span>
              </div>
              {f.extracted_text && <p className="mono text-xs text-muted mb-1">&quot;{f.extracted_text}&quot;</p>}
              <p className="text-xs text-muted mt-1">{f.notes}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card p-5">
        <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">Violations ({scan.violations.length})</p>
        {scan.violations.length === 0 ? (
          <p className="text-sm text-brand-700">No violations detected.</p>
        ) : (
          <div className="space-y-2">
            {scan.violations.map((v: any, i: number) => (
              <div key={i} className="border border-line rounded-md p-3 bg-canvas">
                <div className="flex items-center gap-2 mb-1">
                  <span className="mono text-[10px] font-semibold">{v.severity}</span>
                  <span className="mono text-[10px] text-muted">{v.rule_code}</span>
                </div>
                <p className="text-sm text-ink">{v.description}</p>
                <p className="text-xs text-muted mt-1">Ref: {v.citation}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
