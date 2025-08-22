import { useEffect, useState } from "react";
import { supabase } from "../supabase";

export default function History() {
  const [transcripts, setTranscripts] = useState([]);
  const [editId, setEditId] = useState(null);
  const [editText, setEditText] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTranscripts();
  }, []);

  const fetchTranscripts = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from("transcripts")
        .select("id, transcript_text, word_timestamps, audio_filename, created_at")
        .order("created_at", { ascending: false });

      if (error) {
        throw error;
      }

      setTranscripts(data || []);
    } catch (err) {
      setError("Failed to load transcripts: " + err.message);
      console.error("Error fetching transcripts:", err);
    } finally {
      setLoading(false);
    }
  };

  const saveEdit = async (id) => {
    try {
      const { error } = await supabase
        .from("transcripts")
        .update({ transcript_text: editText })
        .eq("id", id);

      if (error) {
        throw error;
      }

      setEditId(null);
      setEditText("");
      await fetchTranscripts();
    } catch (err) {
      setError("Failed to save edit: " + err.message);
      console.error("Error saving edit:", err);
    }
  };

  const deleteTranscript = async (id) => {
    if (!confirm("Are you sure you want to delete this transcript?")) {
      return;
    }

    try {
      const { error } = await supabase
        .from("transcripts")
        .delete()
        .eq("id", id);

      if (error) {
        throw error;
      }

      await fetchTranscripts();
    } catch (err) {
      setError("Failed to delete transcript: " + err.message);
      console.error("Error deleting transcript:", err);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (wordTimestamps) => {
    if (!wordTimestamps || wordTimestamps.length === 0) return "0s";
    
    try {
      const timestamps = typeof wordTimestamps === 'string' 
        ? JSON.parse(wordTimestamps) 
        : wordTimestamps;
      
      if (timestamps.length > 0) {
        const lastWord = timestamps[timestamps.length - 1];
        return `${lastWord.end}s`;
      }
    } catch (e) {
      console.error("Error parsing word timestamps:", e);
    }
    
    return "0s";
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading transcripts...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
        <strong>Error:</strong> {error}
        <button 
          onClick={() => fetchTranscripts()} 
          className="ml-2 underline hover:no-underline"
        >
          Try again
        </button>
      </div>
    );
  }

  if (transcripts.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p className="text-lg">No transcripts yet</p>
        <p className="text-sm">Record some audio to see your transcript history here!</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-800">Transcript History</h2>
        <button 
          onClick={fetchTranscripts}
          className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded transition-colors"
        >
          🔄 Refresh
        </button>
      </div>

      {transcripts.map((transcript) => (
        <div key={transcript.id} className="border border-gray-200 p-4 rounded-lg bg-white shadow-sm">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
                <span>📅 {formatTimestamp(transcript.created_at)}</span>
                <span>⏱️ {formatDuration(transcript.word_timestamps)}</span>
                {transcript.audio_filename && (
                  <span>🎵 {transcript.audio_filename}</span>
                )}
              </div>

              {editId === transcript.id ? (
                <div className="space-y-3">
                  <textarea
                    className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={4}
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    placeholder="Edit your transcript..."
                  />
                  <div className="flex gap-2">
                    <button 
                      className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
                      onClick={() => saveEdit(transcript.id)}
                    >
                      💾 Save
                    </button>
                    <button 
                      className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
                      onClick={() => {
                        setEditId(null);
                        setEditText("");
                      }}
                    >
                      ❌ Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-gray-800 leading-relaxed">
                      {transcript.transcript_text || "No transcript text available"}
                    </p>
                  </div>

                  {transcript.word_timestamps && (
                    <div>
                      <h4 className="font-medium text-gray-700 mb-2">Words with Timestamps:</h4>
                      <div className="flex flex-wrap gap-2">
                        {(() => {
                          try {
                            const timestamps = typeof transcript.word_timestamps === 'string' 
                              ? JSON.parse(transcript.word_timestamps) 
                              : transcript.word_timestamps;
                            
                            return timestamps.map((word, i) => (
                              <span 
                                key={i} 
                                className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-medium"
                                title={`${word.start}s - ${word.end}s`}
                              >
                                {word.word}
                              </span>
                            ));
                          } catch (e) {
                            return <span className="text-gray-500">Timestamps not available</span>;
                          }
                        })()}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2 pt-2">
                    <button
                      className="px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
                      onClick={() => {
                        setEditId(transcript.id);
                        setEditText(transcript.transcript_text || "");
                      }}
                    >
                      ✏️ Edit
                    </button>
                    <button
                      className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 text-white rounded transition-colors"
                      onClick={() => deleteTranscript(transcript.id)}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}