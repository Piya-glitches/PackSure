import { useEffect, useState } from "react";
import { api } from "../api/client";

interface Sample {
  id: string;
  title: string;
  expected_outcome: string;
}

const PILL: Record<string, string> = {
  COMPLIANT: "pill-pass",
  NON_COMPLIANT: "pill-fail",
  NEEDS_REVIEW: "pill-warn",
};

export default function SampleLabelPicker({ onSelect, loading }: { onSelect: (id: string) => void; loading: boolean }) {
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    api.get("/scans/samples").then((r) => setSamples(r.data.samples));
  }, []);

  return (
    <div className="grid sm:grid-cols-3 gap-3">
      {samples.map((s) => (
        <button
          key={s.id}
          disabled={loading}
          onClick={() => onSelect(s.id)}
          className="card p-4 text-left hover:border-brand-600 transition-colors disabled:opacity-50"
        >
          <span className={`pill ${PILL[s.expected_outcome]} mb-3`}>{s.expected_outcome.replace("_", " ")}</span>
          <p className="text-sm text-ink font-medium">{s.title}</p>
        </button>
      ))}
    </div>
  );
}
