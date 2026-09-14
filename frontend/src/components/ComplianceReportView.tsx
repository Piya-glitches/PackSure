import type { ComplianceReport } from "../types";
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

const SEVERITY_STYLE: Record<string, string> = {
  CRITICAL: "border-red-200 bg-red-50",
  MAJOR: "border-amber-200 bg-amber-50",
  MINOR: "border-line bg-canvas",
};

export default function ComplianceReportView({ report }: { report: ComplianceReport }) {
  const meta = STATUS_META[report.overall_status];

  return (
    <div className="space-y-5">
      <div className="card p-6 flex items-center justify-between">
        <div>
          <h2 className={`text-xl font-semibold ${meta.color}`}>{meta.label}</h2>
          <p className="text-xs text-muted mt-1">PDP detection: {report.pdp_detection_method}</p>
        </div>
        <span className="mono text-3xl text-ink">
          {report.compliance_score}<span className="text-base text-muted">/100</span>
        </span>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <div className="card p-5">
          <p className="text-xs font-medium text-muted uppercase tracking-wide mb-2">Calibration</p>
          {report.calibration.found ? (
            <>
              <p className="text-sm text-ink">
                {report.calibration.symbology} barcode detected{report.calibration.raw_value ? ` (${report.calibration.raw_value})` : ""}
              </p>
              <p className="mono text-xs text-brand-700 mt-1">
                {report.calibration.px_per_mm?.toFixed(2)} px/mm from {report.calibration.physical_width_mm}mm reference
              </p>
            </>
          ) : (
            <p className="text-sm text-amber">No barcode found — font-size checks are unverifiable for this scan.</p>
          )}
        </div>

        <div className="card p-5">
          <p className="text-xs font-medium text-muted uppercase tracking-wide mb-2">Image quality</p>
          <p className="text-sm text-ink">Sharpness variance: <span className="mono">{report.quality_gate.laplacian_variance.toFixed(1)}</span></p>
          <p className="text-sm text-ink">Brightness: <span className="mono">{report.quality_gate.brightness_mean.toFixed(0)}/255</span></p>
          {report.quality_gate.reasons.map((r, i) => (
            <p key={i} className="text-xs text-amber mt-1">{r}</p>
          ))}
        </div>
      </div>

      <div className="card p-5">
        <p className="text-xs font-medium text-muted uppercase tracking-wide mb-3">Responsible-party resolution</p>
        <div className="flex items-center gap-3 mb-2">
          <span className={`pill ${report.responsible_party.is_legally_responsible ? "pill-pass" : "pill-fail"}`}>
            {report.responsible_party.role.replace("_", " ")}
          </span>
          {report.responsible_party.matched_phrase && (
            <span className="mono text-xs text-muted">&quot;{report.responsible_party.matched_phrase}&quot;</span>
          )}
        </div>
        <p className="text-sm text-muted leading-relaxed">{report.responsible_party.reasoning}</p>
      </div>

      <div className="card p-5">
        <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">Mandatory field checklist</p>
        <div className="space-y-3">
          {report.fields.map((f) => (
            <div key={f.field_key} className="border-b border-line pb-3 last:border-0 last:pb-0">
              <div className="flex items-center justify-between mb-1">
                <p className="text-sm text-ink font-medium">{FIELD_LABELS[f.field_key]}</p>
                <span className={`pill ${FIELD_PILL[f.status]}`}>{f.status.replace("_", " ")}</span>
              </div>
              {f.extracted_text && <p className="mono text-xs text-muted mb-1">&quot;{f.extracted_text}&quot;</p>}
              {f.font_height_mm !== null && f.min_required_mm !== null && (
                <p className="mono text-xs text-muted">{f.font_height_mm.toFixed(2)}mm measured / {f.min_required_mm.toFixed(1)}mm required</p>
              )}
              <p className="text-xs text-muted mt-1">{f.notes}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card p-5">
        <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">Violations ({report.violations.length})</p>
        {report.violations.length === 0 ? (
          <p className="text-sm text-brand-700">No violations detected.</p>
        ) : (
          <div className="space-y-2">
            {report.violations.map((v, i) => (
              <div key={i} className={`border rounded-md p-3 ${SEVERITY_STYLE[v.severity]}`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="mono text-[10px] font-semibold text-ink">{v.severity}</span>
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
