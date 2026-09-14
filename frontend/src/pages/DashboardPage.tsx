import { useEffect, useState } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { api } from "../api/client";

interface Stats {
  total: number;
  compliant: number;
  non_compliant: number;
  needs_review: number;
  avg_score: number;
  trend: { date: string; score: number }[];
  violation_breakdown: { rule_code: string; severity: string; count: number }[];
}

const SEVERITY_COLORS: Record<string, string> = { CRITICAL: "#A3402B", MAJOR: "#C97A2B", MINOR: "#9AA3B5" };

function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="card p-5">
      <p className="text-xs font-medium text-muted uppercase tracking-wide mb-2">{label}</p>
      <p className={`mono text-3xl ${color ?? "text-ink"}`}>{value}</p>
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    api.get("/stats").then((r) => setStats(r.data));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-ink mb-1">Enforcement dashboard</h1>
        <p className="text-sm text-muted">Aggregate compliance monitoring across all inspected products.</p>
      </div>

      {!stats ? (
        <p className="text-sm text-muted">Loading…</p>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <StatCard label="Total scans" value={stats.total} />
            <StatCard label="Compliant" value={stats.compliant} color="text-brand-700" />
            <StatCard label="Non-compliant" value={stats.non_compliant} color="text-red-600" />
            <StatCard label="Needs review" value={stats.needs_review} color="text-amber" />
            <StatCard label="Avg score" value={stats.avg_score} />
          </div>

          {stats.total === 0 ? (
            <div className="card p-10 text-center">
              <p className="text-sm text-muted">No scans recorded yet. Run and save a scan to populate this dashboard.</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 gap-6">
              <div className="card p-5">
                <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">Compliance score over time</p>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={stats.trend}>
                    <CartesianGrid stroke="#E4E7E3" strokeDasharray="3 3" />
                    <XAxis dataKey="date" stroke="#5B6560" fontSize={10} tickLine={false} />
                    <YAxis domain={[0, 100]} stroke="#5B6560" fontSize={10} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E4E7E3", fontSize: 12, borderRadius: 8 }} />
                    <Line type="monotone" dataKey="score" stroke="#2F6B4F" strokeWidth={2} dot={{ r: 3, fill: "#2F6B4F" }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="card p-5">
                <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">Top violation types</p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={stats.violation_breakdown} layout="vertical" margin={{ left: 12 }}>
                    <CartesianGrid stroke="#E4E7E3" strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" stroke="#5B6560" fontSize={10} tickLine={false} />
                    <YAxis dataKey="rule_code" type="category" stroke="#5B6560" fontSize={9} width={150} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E4E7E3", fontSize: 12, borderRadius: 8 }} />
                    <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                      {stats.violation_breakdown.map((d, i) => (
                        <Cell key={i} fill={SEVERITY_COLORS[d.severity] ?? "#9AA3B5"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
