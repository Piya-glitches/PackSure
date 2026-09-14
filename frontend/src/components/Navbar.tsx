import { Link, useLocation, useNavigate } from "react-router-dom";
import { getAuthState, clearAuthState } from "../api/client";

export default function Navbar() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const auth = getAuthState();

  const links = [
    { to: "/scan", label: "Scan" },
    { to: "/history", label: "History" },
    ...(auth.role === "officer" || auth.role === "admin" ? [{ to: "/dashboard", label: "Dashboard" }] : []),
  ];

  return (
    <header className="border-b border-line bg-white sticky top-0 z-40">
      <div className="mx-auto max-w-6xl px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5">
          <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
            <rect x="1" y="1" width="24" height="24" rx="6" fill="#2F6B4F" />
            <path d="M7 13.5L11 17.5L19 8.5" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-lg font-semibold tracking-tight text-ink">PackSure</span>
        </Link>

        <nav className="flex items-center gap-1">
          {links.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={`px-3.5 py-1.5 text-sm rounded-md transition-colors ${
                pathname.startsWith(l.to) ? "bg-brand-100 text-brand-700 font-medium" : "text-muted hover:text-ink"
              }`}
            >
              {l.label}
            </Link>
          ))}

          {auth.token ? (
            <div className="flex items-center gap-3 ml-3 pl-3 border-l border-line">
              <span className="text-xs text-muted">
                {auth.username} · <span className="mono">{auth.role}</span>
              </span>
              <button
                onClick={() => {
                  clearAuthState();
                  navigate("/");
                }}
                className="text-xs text-muted hover:text-ink"
              >
                Sign out
              </button>
            </div>
          ) : (
            <Link to="/login" className="ml-3 text-sm text-brand-700 font-medium hover:underline">
              Sign in
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}
