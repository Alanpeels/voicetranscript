import { createClient } from "@supabase/supabase-js";

// These values come from your Supabase project settings
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Create the client
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
