import os
import io
import gc
import logging
import traceback
from datetime import datetime
import sqlite3
import threading
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# Fixed CORS configuration
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
     supports_credentials=False)

# Manual CORS headers as backup
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

# -----------------------------
# Global model variables
# -----------------------------
processor = None
model = None
model_loaded = False
model_error = None
model_loading = False  # Prevent concurrent loading
model_load_lock = threading.Lock()

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
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Additional cleanup
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)

def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        logger.info(f"System memory: {mem.percent}% used, {mem.available/1024**3:.2f}GB available")
        logger.info(f"Process memory: {process.memory_info().rss/1024**2:.1f}MB")
    except ImportError:
        logger.info("psutil not available for memory logging")

def load_model():
    global processor, model, model_loaded, model_error, model_loading
    
    with model_load_lock:  # Thread-safe loading
        if model_loaded:
            return True
        
        if model_loading:
            # Wait for another thread to finish loading
            while model_loading and not model_loaded:
                time.sleep(0.1)
            return model_loaded
        
        model_loading = True
        
        try:
            logger.info("Loading Wav2Vec2 model...")
            log_memory_usage()
            
            model_name = "facebook/wav2vec2-base-960h"
            
            # Load processor first (smaller)
            logger.info("Loading processor...")
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            clear_memory()
            
            # Load model
            logger.info("Loading model...")
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            model.eval()
            
            # Ensure CPU usage
            model = model.to("cpu")
            
            # Set model to use less memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            model_loaded = True
            model_loading = False
            
            logger.info("Model loaded successfully")
            log_memory_usage()
            clear_memory()
            
            return True
            
        except Exception as e:
            model_error = str(e)
            model_loading = False
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            log_memory_usage()
            return False

def process_audio(audio_bytes, max_duration=30):  # Increased from 15 to 30 seconds
    try:
        logger.info(f"Processing audio: {len(audio_bytes)} bytes")
        
        # More robust audio loading
        audio_array, sr = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=16000, 
            duration=max_duration,
            res_type='kaiser_fast'  # Faster resampling
        )
        
        logger.info(f"Audio processed: {len(audio_array)} samples at {sr}Hz, duration: {len(audio_array)/sr:.2f}s")
        return audio_array, sr
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise Exception(f"Could not process audio file: {str(e)}")

def transcribe(audio_array, sr):
    if not load_model():
        raise Exception(f"Model load failed: {model_error}")

    try:
        logger.info("Starting transcription...")
        log_memory_usage()
        
        # Process audio with error handling
        inputs = processor(
            audio_array, 
            sampling_rate=sr, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=250000  # Limit input length
        )
        
        logger.info("Running model inference...")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Get predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_text = processor.batch_decode(predicted_ids)[0].strip()
        
        logger.info(f"Raw transcript: '{transcript_text}'")
        
        # Handle empty transcription
        if not transcript_text:
            transcript_text = "[No speech detected]"
            words = []
        else:
            words = transcript_text.split()
        
        # Create word timestamps
        word_timestamps = []
        total_duration = len(audio_array) / sr
        
        if len(words) > 0:
            words_per_sec = len(words) / total_duration if total_duration > 0 else 0
            for i, word in enumerate(words):
                start = i / words_per_sec if words_per_sec > 0 else 0
                end = (i + 1) / words_per_sec if words_per_sec > 0 else 0
                word_timestamps.append({
                    "word": word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": 0.8  # Placeholder
                })
        
        # Cleanup
        del inputs, logits, predicted_ids
        clear_memory()
        
        logger.info(f"Transcription completed: {len(words)} words")
        return transcript_text, word_timestamps
        
    except Exception as e:
        clear_memory()
        logger.error(f"Transcription failed: {e}")
        logger.error(traceback.format_exc())
        raise Exception(f"Transcription failed: {str(e)}")

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Wav2Vec2 ASR API running",
        "endpoints": ["/transcribe", "/health", "/models"],
        "status": "ready",
        "model": "wav2vec2-base-960h"
    })

@app.route("/health", methods=["GET"])
def health_check():
    try:
        status = {
            "status": "healthy",
            "model": "wav2vec2-base-960h",
            "model_loaded": model_loaded,
            "model_loading": model_loading
        }
        
        if model_error:
            status["model_error"] = model_error
        
        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            process = psutil.Process()
            status.update({
                "memory_usage": f"{mem.percent}%",
                "available_memory": f"{mem.available / 1024**3:.2f} GB",
                "process_memory": f"{process.memory_info().rss/1024**2:.1f}MB"
            })
        except ImportError:
            status["memory_info"] = "unavailable"
            
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy", 
            "error": str(e)
        }), 500

@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    try:
        logger.info("Received transcription request")
        
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        if not audio_file.filename:
            return jsonify({"error": "Empty audio file"}), 400
            
        audio_bytes = audio_file.read()
        
        # File size check
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio file"}), 400
            
        if len(audio_bytes) > 10 * 1024 * 1024:  # Increased to 10MB
            return jsonify({"error": "Audio file too large (max 10MB)"}), 400

        logger.info(f"Processing file: {audio_file.filename}, size: {len(audio_bytes)} bytes")

        # Process audio
        audio_array, sr = process_audio(audio_bytes)
        
        # Transcribe
        transcript, word_timestamps = transcribe(audio_array, sr)

        # Generate response
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"

        # Optional: save to database
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcripts (audio_filename, transcript_text, word_timestamps, model_used) VALUES (?, ?, ?, ?)",
                (audio_filename, transcript, str(word_timestamps), "wav2vec2-base-960h")
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to save transcript: {db_err}")

        # Cleanup
        del audio_bytes, audio_array
        clear_memory()

        response = {
            "success": True,
            "transcript": transcript,
            "word_timestamps": word_timestamps,
            "audio_filename": audio_filename,
            "model_used": "wav2vec2-base-960h",
            "word_count": len(word_timestamps),
            "duration": len(word_timestamps) * 0.5 if word_timestamps else 0
        }
        
        logger.info(f"Transcription successful: '{transcript[:50]}...'")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Transcription route error: {e}")
        logger.error(traceback.format_exc())
        clear_memory()
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route("/models", methods=["GET"])
def models_route():
    return jsonify({
        "current_model": "wav2vec2-base-960h",
        "size": "~360MB",
        "language": "English",
        "features": ["word-level transcription", "fast inference", "CPU optimized"],
        "limitations": ["English only", "Best for clear speech"]
    })

@app.route("/preload", methods=["POST"])
def preload_model():
    """Endpoint to preload the model"""
    try:
        success = load_model()
        if success:
            return jsonify({"message": "Model preloaded successfully", "model_loaded": model_loaded})
        else:
            return jsonify({"error": f"Model preload failed: {model_error}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Error handlers
# -----------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        init_db()
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Starting server on port {port}")
        
        # Optional: preload model on startup (comment out if causing timeouts)
        # logger.info("Preloading model...")
        # load_model()
        
        app.run(host="0.0.0.0", port=port, threaded=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)