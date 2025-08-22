import os
import io
import gc
import logging
import traceback
from datetime import datetime
import sqlite3

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)

# -----------------------------
# Global model variables
# -----------------------------
processor = None
model = None
model_loaded = False
model_error = None

# -----------------------------
# Database
# -----------------------------
DB_FILE = "transcripts.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_filename TEXT,
                transcript_text TEXT,
                word_timestamps TEXT,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# -----------------------------
# Utility functions
# -----------------------------
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model():
    global processor, model, model_loaded, model_error
    if model_loaded:
        return True
    try:
        logger.info("Loading Wav2Vec2 model...")
        model_name = "facebook/wav2vec2-base-960h"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.eval()
        model = model.to("cpu")  # Free tier: use CPU
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        model_error = str(e)
        logger.error(f"Failed to load model: {e}")
        return False

def process_audio(audio_bytes, max_duration=15):
    try:
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=max_duration)
        return audio_array, sr
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise

def transcribe(audio_array, sr):
    if not load_model():
        raise Exception(f"Model load failed: {model_error}")

    try:
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_text = processor.batch_decode(predicted_ids)[0]

        # Simple word timestamps approximation
        words = transcript_text.split()
        word_timestamps = []
        total_duration = len(audio_array) / sr
        if len(words) > 0:
            words_per_sec = len(words) / total_duration
            for i, word in enumerate(words):
                start = i / words_per_sec
                end = (i + 1) / words_per_sec
                word_timestamps.append({
                    "word": word,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "confidence": 0.8
                })

        del inputs, logits, predicted_ids
        clear_memory()

        return transcript_text, word_timestamps
    except Exception as e:
        clear_memory()
        logger.error(f"Transcription failed: {e}")
        raise

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Wav2Vec2 ASR API running",
        "endpoints": ["/transcribe", "/health", "/models"]
    })

@app.route("/health", methods=["GET"])
def health_check():
    try:
        import psutil
        mem = psutil.virtual_memory()
        return jsonify({
            "status": "healthy",
            "model": "wav2vec2-base-960h",
            "model_loaded": model_loaded,
            "memory_usage": f"{mem.percent}%",
            "available_memory": f"{mem.available / 1024**3:.2f} GB",
            "model_error": model_error
        })
    except ImportError:
        return jsonify({"status": "healthy", "model_loaded": model_loaded})

@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()

        if len(audio_bytes) > 5 * 1024 * 1024:
            return jsonify({"error": "Audio file too large (max 5MB)"}), 400

        audio_array, sr = process_audio(audio_bytes)
        transcript, word_timestamps = transcribe(audio_array, sr)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"

        # Optional: save to SQLite
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcripts (audio_filename, transcript_text, word_timestamps, model_used) VALUES (?, ?, ?, ?)",
                (audio_filename, transcript, str(word_timestamps), "wav2vec2-base-960h")
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to save transcript: {db_err}")

        return jsonify({
            "success": True,
            "transcript": transcript,
            "word_timestamps": word_timestamps,
            "audio_filename": audio_filename,
            "model_used": "wav2vec2-base-960h",
            "word_count": len(word_timestamps)
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        clear_memory()
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def models_route():
    return jsonify({
        "current_model": "wav2vec2-base-960h",
        "size": "~360MB",
        "language": "English",
        "features": ["word-level transcription", "fast inference"]
    })

# -----------------------------
# Error handlers
# -----------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
