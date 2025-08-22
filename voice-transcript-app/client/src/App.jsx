import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./hooks/useAuth";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";

export default function App() {
  const user = useAuth();

  if (user === undefined) return <p className="p-8">Loading…</p>;

  return (
    <Routes>
      <Route path="/" element={user ? <Dashboard /> : <Navigate to="/auth" />} />
      <Route path="/auth" element={!user ? <Auth /> : <Navigate to="/" />} />
    </Routes>
  );
}