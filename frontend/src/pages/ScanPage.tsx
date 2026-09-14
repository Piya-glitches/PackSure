import { useState } from "react";
import Uploader from "../components/Uploader";
import SampleLabelPicker from "../components/SampleLabelPicker";
import ImageWithOverlay from "../components/ImageWithOverlay";
import ComplianceReportView from "../components/ComplianceReportView";
import { api } from "../api/client";
import type { ComplianceReport } from "../types";

export default function ScanPage() {
  const [productLabel, setProductLabel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ComplianceReport | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function runFromDataUrl(dataUrl: string) {
    setLoading(true);
    setError(null);
    setReport(null);
    setSavedId(null);
    try {
      const res = await api.post("/scans/run", { product_label: productLabel || "Unnamed scan", image_data_url: dataUrl });
      setReport(res.data.report);
      setProcessedImage(res.data.processed_image_data_url);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Pipeline failed. The backend may still be downloading model weights on first run — try again in a moment.");
    } finally {
      setLoading(false);
    }
  }

  async function runSample(id: string) {
    setLoading(true);
    setError(null);
    setReport(null);
    setSavedId(null);
    if (!productLabel) setProductLabel(id.replace(/-/g, " "));
    try {
      const res = await api.get(`/scans/samples/${id}/run`);
      setReport(res.data.report);
      setProcessedImage(res.data.processed_image_data_url);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Failed to run sample.");
    } finally {
      setLoading(false);
    }
  }

  async function saveScan() {
    if (!report || !processedImage) return;
    setSaving(true);
    try {
      const res = await api.post("/scans", {
        product_label: productLabel || "Unnamed scan",
        processed_image_data_url: processedImage,
        report,
      });
      setSavedId(res.data.id);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Failed to save scan.");
    } finally {
      setSaving(false);
    }
  }

  function reset() {
    setReport(null);
    setProcessedImage(null);
    setSavedId(null);
    setError(null);
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-ink mb-1">Scan a label</h1>
      <p className="text-sm text-muted mb-8">Upload a photo, or try a built-in sample to see the full pipeline run end to end.</p>

      {error && <div className="mb-6 border border-red-200 bg-red-50 rounded-md p-3 text-sm text-red-700">{error}</div>}

      {!report && !loading && (
        <div className="space-y-8">
          <div>
            <label className="text-xs font-medium text-muted block mb-2 uppercase tracking-wide">Product nickname (optional)</label>
            <input
              type="text"
              value={productLabel}
              onChange={(e) => setProductLabel(e.target.value)}
              placeholder="e.g. Amul Butter 500g"
              className="max-w-sm"
            />
          </div>
          <Uploader onImage={runFromDataUrl} />
          <div>
            <p className="text-xs font-medium text-muted mb-3 uppercase tracking-wide">Or try a built-in sample label</p>
            <SampleLabelPicker onSelect={runSample} loading={loading} />
          </div>
        </div>
      )}

      {loading && (
        <div className="card p-10 text-center">
          <p className="text-sm text-ink mb-1">Running the pipeline…</p>
          <p className="text-xs text-muted">Barcode calibration → PDP detection → dewarping → OCR → field classification → rule validation</p>
        </div>
      )}

      {report && processedImage && (
        <div className="grid lg:grid-cols-[380px_1fr] gap-8">
          <div className="space-y-4">
            <ImageWithOverlay imageDataUrl={processedImage} report={report} />
            <div className="flex gap-2">
              <button onClick={saveScan} disabled={saving || !!savedId} className="btn-primary flex-1">
                {savedId ? "Saved ✓" : saving ? "Saving…" : "Save to history"}
              </button>
              {savedId && (
                <a href={`${api.defaults.baseURL}/scans/${savedId}/report`} target="_blank" rel="noreferrer" className="btn-secondary">
                  PDF
                </a>
              )}
            </div>
            <button onClick={reset} className="btn-secondary w-full">Scan another label</button>
          </div>
          <ComplianceReportView report={report} />
        </div>
      )}
    </div>
  );
}
