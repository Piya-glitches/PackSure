import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { api, setAuthState } from "../api/client";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.post(`/auth/login?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`);
      setAuthState(res.data.access_token, res.data.role, username);
      navigate("/scan");
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-sm px-6 py-20">
      <h1 className="text-2xl font-semibold text-ink mb-1">Sign in</h1>
      <p className="text-sm text-muted mb-8">Officers and admins get access to the enforcement dashboard.</p>

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
        <button type="submit" disabled={loading} className="btn-primary w-full">
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="text-sm text-muted mt-4">
        No account? <Link to="/register" className="text-brand-700 font-medium">Register</Link>
      </p>
      <p className="text-sm text-muted mt-6">
        Or continue as a <Link to="/scan" className="text-brand-700 font-medium">public/anonymous user</Link> — scanning doesn't require an account.
      </p>
    </div>
  );
}
