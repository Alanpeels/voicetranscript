import { useState } from "react";
import { supabase } from "../supabase";

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handle = async (e) => {
    e.preventDefault();
    const { error } = isLogin
      ? await supabase.auth.signInWithPassword({ email, password })
      : await supabase.auth.signUp({ email, password });
    if (error) alert(error.message);
  };

  return (
    <div className="max-w-sm mx-auto mt-20 p-6 shadow rounded space-y-4">
      <h1 className="text-xl font-bold">{isLogin ? "Log in" : "Sign up"}</h1>
      <form onSubmit={handle} className="space-y-3">
        <input
          className="input"
          placeholder="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          className="input"
          placeholder="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button className="btn-primary w-full">{isLogin ? "Log in" : "Sign up"}</button>
      </form>
      <button
        className="text-sm text-blue-600"
        onClick={() => setIsLogin(!isLogin)}
      >
        {isLogin ? "Need an account?" : "Already have an account?"}
      </button>
    </div>
  );
}