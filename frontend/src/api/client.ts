import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = axios.create({ baseURL: API_URL });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("packsure_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export interface AuthState {
  token: string | null;
  role: string | null;
  username: string | null;
}

export function getAuthState(): AuthState {
  return {
    token: localStorage.getItem("packsure_token"),
    role: localStorage.getItem("packsure_role"),
    username: localStorage.getItem("packsure_username"),
  };
}

export function setAuthState(token: string, role: string, username: string) {
  localStorage.setItem("packsure_token", token);
  localStorage.setItem("packsure_role", role);
  localStorage.setItem("packsure_username", username);
}

export function clearAuthState() {
  localStorage.removeItem("packsure_token");
  localStorage.removeItem("packsure_role");
  localStorage.removeItem("packsure_username");
}
