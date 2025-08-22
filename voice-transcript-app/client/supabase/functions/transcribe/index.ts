import { serve } from "std/http/server.ts";
import { createClient } from "@supabase/supabase-js";

// Create a Supabase client
const supabaseClient = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_ANON_KEY")!
);

// Define the Whisper backend URL
const WHISPER_BACKEND_URL = Deno.env.get("WHISPER_BACKEND_URL") || "http://localhost:5000";

// Define the function handler
serve(async (req: Request) => {
  // Check if the request is POST
  if (req.method === "POST") {
    try {
      // Get the form data (audio file)
      const formData = await req.formData();
      const audioFile = formData.get("audio") as File;
      
      if (!audioFile) {
        return new Response(
          JSON.stringify({ error: "No audio file provided" }), 
          { 
            status: 400,
            headers: { "Content-Type": "application/json" }
          }
        );
      }

      // Convert audio file to buffer
      const audioBuffer = await audioFile.arrayBuffer();
      
      // Create new form data for the Whisper backend
      const whisperFormData = new FormData();
      const audioBlob = new Blob([audioBuffer], { type: audioFile.type });
      whisperFormData.append("audio", audioBlob, audioFile.name);

      // Send to Whisper backend
      const whisperResponse = await fetch(`${WHISPER_BACKEND_URL}/transcribe`, {
        method: "POST",
        body: whisperFormData,
      });

      if (!whisperResponse.ok) {
        const errorText = await whisperResponse.text();
        console.error("Whisper backend error:", errorText);
        return new Response(
          JSON.stringify({ error: "Transcription failed", details: errorText }), 
          { 
            status: 500,
            headers: { "Content-Type": "application/json" }
          }
        );
      }

      // Get transcription result
      const transcriptionData = await whisperResponse.json();
      
      // Store in Supabase database if user is authenticated
      const authHeader = req.headers.get("Authorization");
      if (authHeader && authHeader.startsWith("Bearer ")) {
        const token = authHeader.substring(7);
        
        try {
          // Get user from token
          const { data: { user }, error: userError } = await supabaseClient.auth.getUser(token);
          
          if (user && !userError) {
            // Store transcript in database
            const { error: insertError } = await supabaseClient
              .from("transcripts")
              .insert({
                user_id: user.id,
                audio_filename: transcriptionData.audio_filename,
                transcript_text: transcriptionData.transcript,
                word_timestamps: JSON.stringify(transcriptionData.word_timestamps),
                created_at: new Date().toISOString()
              });
            
            if (insertError) {
              console.error("Database insert error:", insertError);
            }
          }
        } catch (dbError) {
          console.error("Database operation failed:", dbError);
          // Continue without storing in DB if there's an error
        }
      }

      // Return the transcription data
      return new Response(
        JSON.stringify(transcriptionData),
        { 
          headers: { "Content-Type": "application/json" },
          status: 200
        }
      );
      
    } catch (error) {
      console.error("Function error:", error);
      return new Response(
        JSON.stringify({ error: "Internal server error", details: error.message }), 
        { 
          status: 500,
          headers: { "Content-Type": "application/json" }
        }
      );
    }
  } else {
    // If not POST, return a 405 Method Not Allowed
    return new Response(
      JSON.stringify({ error: "Method not allowed" }), 
      { 
        status: 405,
        headers: { "Content-Type": "application/json" }
      }
    );
  }
});