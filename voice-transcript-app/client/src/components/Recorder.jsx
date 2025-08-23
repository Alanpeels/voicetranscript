import { useState, useRef } from "react";
import { supabase } from "../supabase";
import { config } from "../config";

export default function Recorder() {
  const [status, setStatus] = useState("idle");
  const [transcript, setTranscript] = useState(null);
  const [wordTimestamps, setWordTimestamps] = useState([]);
  const [error, setError] = useState(null);
  const mediaRef = useRef(null);
  const audioChunks = useRef([]);

  const WHISPER_BACKEND_URL = config.backendUrl;

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mimeType = "audio/webm;codecs=opus";
      mediaRef.current = new MediaRecorder(stream, { mimeType });
      audioChunks.current = [];
      mediaRef.current.ondataavailable = (e) => audioChunks.current.push(e.data);
      mediaRef.current.start();
      setStatus("recording");
      console.log(`Recording started with format: ${mimeType}`);
    } catch (err) {
      setError("Failed to start recording: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRef.current && mediaRef.current.state === "recording") {
      mediaRef.current.stop();
      mediaRef.current.onstop = uploadAudio;
      setStatus("processing");
    }
  };

  const convertWebmToWav = async (webmBlob) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      const arrayBuffer = await webmBlob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      const targetSampleRate = 16000;
      const targetBuffer = audioContext.createBuffer(
        audioBuffer.numberOfChannels,
        Math.round(audioBuffer.length * targetSampleRate / audioBuffer.sampleRate),
        targetSampleRate
      );
      
      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const sourceData = audioBuffer.getChannelData(channel);
        const targetData = targetBuffer.getChannelData(channel);
        
        for (let i = 0; i < targetData.length; i++) {
          const sourceIndex = (i * audioBuffer.sampleRate) / targetSampleRate;
          const sourceIndexFloor = Math.floor(sourceIndex);
          const sourceIndexCeil = Math.min(sourceIndexFloor + 1, sourceData.length - 1);
          const fraction = sourceIndex - sourceIndexFloor;
          
          targetData[i] = sourceData[sourceIndexFloor] * (1 - fraction) + sourceData[sourceIndexCeil] * fraction;
        }
      }
      
      const wavBlob = audioBufferToWav(targetBuffer);
      return wavBlob;
      
    } catch (error) {
      console.error("Audio conversion failed:", error);
      return webmBlob;
    }
  };

  const audioBufferToWav = (buffer) => {
    const length = buffer.length;
    const numberOfChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
    const view = new DataView(arrayBuffer);
    
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * numberOfChannels * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numberOfChannels * 2, true);
    view.setUint16(32, numberOfChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * numberOfChannels * 2, true);
    
    let offset = 44;
    for (let i = 0; i < length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  };

  const uploadAudio = async () => {
    try {
      const webmBlob = new Blob(audioChunks.current, { type: "audio/webm" });
      
      console.log("Converting WebM to WAV...");
      const wavBlob = await convertWebmToWav(webmBlob);
      console.log("Conversion completed, WAV size:", wavBlob.size);
      
      const formData = new FormData();
      formData.append("audio", wavBlob, `rec_${Date.now()}.wav`);

      const response = await fetch(`${WHISPER_BACKEND_URL}/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();

      if (data.success) {
        setTranscript(data.transcript);
        setWordTimestamps(data.word_timestamps || []);
        
        try {
          const { data: { user } } = await supabase.auth.getUser();
          if (user) {
            await supabase.from("transcripts").insert({
              user_id: user.id,
              audio_filename: data.audio_filename,
              transcript_text: data.transcript,
              word_timestamps: JSON.stringify(data.word_timestamps),
              created_at: new Date().toISOString()
            });
          }
        } catch (dbError) {
          console.error("Database storage failed:", dbError);
        }
      } else {
        throw new Error(data.error || "Transcription failed");
      }

      setStatus("idle");
    } catch (err) {
      setError("Transcription failed: " + err.message);
      setStatus("idle");
    }
  };

  const resetRecording = () => {
    setTranscript(null);
    setWordTimestamps([]);
    setError(null);
    setStatus("idle");
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button
          disabled={status !== "idle"}
          className={`px-4 py-2 rounded font-medium transition-colors ${
            status === "idle" 
              ? "bg-blue-500 hover:bg-blue-600 text-white" 
              : "bg-gray-300 text-gray-500 cursor-not-allowed"
          }`}
          onClick={startRecording}
        >
          {status === "idle" ? "🎤 Add Recording" : "⏸️ Recording..."}
        </button>
        
        <button
          disabled={status !== "recording"}
          className={`px-4 py-2 rounded font-medium transition-colors ${
            status === "recording" 
              ? "bg-red-500 hover:bg-red-600 text-white" 
              : "bg-gray-300 text-gray-500 cursor-not-allowed"
          }`}
          onClick={stopRecording}
        >
          {status === "idle" ? "⏹️ Stop Recording" : "⏹️ Stop"}
        </button>

        {transcript && (
          <button
            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded font-medium transition-colors"
            onClick={resetRecording}
        >
            🔄 New Recording
          </button>
        )}
      </div>

      {status === "processing" && (
        <div className="flex items-center gap-2 text-blue-600">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          <p>Transcribing audio...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {transcript && (
        <div className="border border-gray-200 p-4 rounded-lg bg-white shadow-sm">
          <h3 className="font-semibold mb-3 text-lg text-gray-800">Word-level Transcript</h3>
          
          <div className="mb-4 p-3 bg-gray-50 rounded">
            <p className="text-gray-700">{transcript}</p>
          </div>

          {wordTimestamps.length > 0 && (
            <div>
              <h4 className="font-medium mb-2 text-gray-700">Words with Timestamps:</h4>
              <div className="flex flex-wrap gap-2">
                {wordTimestamps.map((word, i) => (
                  <span 
                    key={i} 
                    className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-medium"
                    title={`${word.start}s - ${word.end}s`}
                  >
                    {word.word}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}