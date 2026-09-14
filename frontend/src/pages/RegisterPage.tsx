import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { api } from "../api/client";

export default function RegisterPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("public");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await api.post("/auth/register", { username, password, role });
      navigate("/login");
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Registration failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-sm px-6 py-20">
      <h1 className="text-2xl font-semibold text-ink mb-1">Create an account</h1>
      <p className="text-sm text-muted mb-8">Officer/admin roles are typically provisioned by an administrator in production; open here for demo purposes.</p>

      <form onSubmit={handleSubmit} className="card p-6 space-y-4">
        {error && <p className="text-sm text-red-600 bg-red-50 p-2 rounded-md">{error}</p>}
        <div>
          <label className="text-xs font-medium text-muted block mb-1">Username</label>
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </div>
        <div>
          <label className="text-xs font-medium text-muted block mb-1">Password</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        <div>
          <label className="text-xs font-medium text-muted block mb-1">Role</label>
          <select value={role} onChange={(e) => setRole(e.target.value)} className="w-full border border-line rounded-md px-3 py-2">
            <option value="public">Public / Consumer</option>
            <option value="officer">Enforcement Officer</option>
            <option value="admin">Admin</option>
          </select>
        </div>
        <button type="submit" disabled={loading} className="btn-primary w-full">
          {loading ? "Creating…" : "Create account"}
        </button>
      </form>

      <p className="text-sm text-muted mt-4">
        Already have an account? <Link to="/login" className="text-brand-700 font-medium">Sign in</Link>
      </p>
    </div>
  );
}
