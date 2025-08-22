// Configuration for different environments
export const config = {
  // Backend API URL - Update this with your Render backend URL after deployment
  backendUrl: import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000',
  
  // Supabase configuration
  supabaseUrl: import.meta.env.VITE_SUPABASE_URL,
  supabaseAnonKey: import.meta.env.VITE_SUPABASE_ANON_KEY,
};
