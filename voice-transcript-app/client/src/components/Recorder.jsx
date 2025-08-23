import React, { useRef, useState } from "react";
import { supabase } from "../supabaseClient";
import { WHISPER_BACKEND_URL } from "../config";

export default function Recorder() {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("idle"); // idle, recording, transcribing, loading_model
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const [wordTimestamps, setWordTimestamps] = useState([]);

  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

  const startRecording = async () => {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorderRef.current.start();
      setRecording(true);
      setStatus("recording");
    } catch (err) {
      setError("Microphone access denied.");
      console.error(err);
    }
  };

  const stopRecording = async () => {
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.onstop = () => {
      setRecording(false);
      setStatus("transcribing");
      uploadAudio();
    };
  };

  const convertWebmToWav = async (webmBlob) => {
    const audioCtx = new AudioContext();
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: "audio/wav" });
  };

  const audioBufferToWav = (buffer) => {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const bufferOut = new ArrayBuffer(length);
    const view = new DataView(bufferOut);

    writeUTFBytes(view, 0, "RIFF");
    view.setUint32(4, 36 + buffer.length * numOfChan * 2, true);
    writeUTFBytes(view, 8, "WAVE");
    writeUTFBytes(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChan, true);
    view.setUint32(24, buffer.sampleRate, true);
    view.setUint32(28, buffer.sampleRate * numOfChan * 2, true);
    view.setUint16(32, numOfChan * 2, true);
    view.setUint16(34, 16, true);
    writeUTFBytes(view, 36, "data");
    view.setUint32(40, buffer.length * numOfChan * 2, true);

    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
      for (let channel = 0; channel < numOfChan; channel++) {
        let sample = buffer.getChannelData(channel)[i];
        let clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clamped * 0x7fff, true);
        offset += 2;
      }
    }
    return bufferOut;
  };

  const writeUTFBytes = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
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

      // ✅ Handle 202 model loading
      if (response.status === 202) {
        const data = await response.json();
        if (data.status === "model_loading") {
          console.log("Model is loading, please wait...");
          setStatus("loading_model");
          setTimeout(() => {
            uploadAudio(); // retry after model loads
          }, (data.retry_after_seconds || 5) * 1000);
        }
        return;
      }

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
              created_at: new Date().toISOString(),
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

  return (
    <div className="p-4 flex flex-col gap-4">
      <div className="flex gap-2">
        {!recording ? (
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded"
            onClick={startRecording}
          >
            Start Recording
          </button>
        ) : (
          <button
            className="px-4 py-2 bg-red-600 text-white rounded"
            onClick={stopRecording}
          >
            Stop Recording
          </button>
        )}
      </div>

      {/* ✅ Status messages */}
      {status === "loading_model" && (
        <div className="flex items-center gap-2 text-yellow-600">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600"></div>
          <p>Model is loading, please wait...</p>
        </div>
      )}
      {status === "transcribing" && <p className="text-blue-600">Transcribing...</p>}
      {status === "recording" && <p className="text-red-600">Recording...</p>}

      {error && <p className="text-red-600">{error}</p>}
      {transcript && (
        <div className="bg-gray-100 p-2 rounded">
          <h3 className="font-bold">Transcript:</h3>
          <p>{transcript}</p>
        </div>
      )}
    </div>
  );
}
