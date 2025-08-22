from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import io
import logging
import traceback
import gc
import os
from datetime import datetime
import sqlite3

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=False)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add(
            'Access-Control-Allow-Headers',
            'Content-Type,Authorization,Accept,Origin,X-Requested-With'
        )
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add(
        'Access-Control-Allow-Headers',
        'Content-Type,Authorization,Accept,Origin,X-Requested-With'
    )
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# =========================
# Global Model Variables
# =========================
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = None
model = None
model_loaded = False
model_error = None

# =========================
# Utility Functions
# =========================
def load_wav2vec2_model():
    global processor, model, model_loaded, model_error
    if model_loaded:
        return True
    try:
        logger.info(f"Loading model: {MODEL_NAME} ...")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        model.eval()
        model = model.to('cpu')
        model_loaded = True
        logger.info(f"Model {MODEL_NAME} loaded successfully")
        return True
    except Exception as e:
        model_error = str(e)
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        return False

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_audio(audio_bytes, max_duration=30):
    """Load audio and convert to 16kHz"""
    try:
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=max_duration)
        logger.info(f"Audio loaded: {len(audio_array)} samples at {sr}Hz")
        return audio_array, sr
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise Exception("Could not process audio")

def transcribe(audio_array, sample_rate):
    if not load_wav2vec2_model():
        raise Exception(f"Model load failed: {model_error}")
    try:
        # Preprocess
        inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Estimate simple word-level timestamps
        hop_length = 320
        time_per_frame = hop_length / sample_rate
        words = transcription.split()
        word_timestamps = []
        if words:
            frame_count = logits.shape[1]
            total_duration = frame_count * time_per_frame
            words_per_second = len(words) / total_duration if total_duration > 0 else 0
            for i, word in enumerate(words):
                start_time = i / words_per_second if words_per_second > 0 else 0
                end_time = (i + 1) / words_per_second if words_per_second > 0 else 0
                word_timestamps.append({
                    "word": word,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "confidence": 0.8
                })
        del inputs, logits, predicted_ids
        clear_memory()
        return transcription, word_timestamps
    except Exception as e:
        clear_memory()
        logger.error(f"Transcription failed: {e}")
        raise

# =========================
# Database
# =========================
DB_FILE = 'transcripts.db'

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
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

# =========================
# Routes
# =========================
@app.route('/transcribe', methods=['POST'])
def transcribe_route():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        if len(audio_bytes) > 5 * 1024 * 1024:
            return jsonify({'error': 'Audio file too large (max 5MB)'}), 400
        audio_array, sr = process_audio(audio_bytes, max_duration=30)
        transcript, word_timestamps = transcribe(audio_array, sr)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"
        
        # Store in DB
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transcripts (audio_filename, transcript_text, word_timestamps, model_used)
                VALUES (?, ?, ?, ?)
            ''', (audio_filename, transcript, str(word_timestamps), MODEL_NAME))
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"DB insert failed: {db_err}")
        
        del audio_bytes, audio_array
        clear_memory()
        
        return jsonify({
            'success': True,
            'transcript': transcript,
            'word_timestamps': word_timestamps,
            'audio_filename': audio_filename,
            'model_used': MODEL_NAME,
            'word_count': len(word_timestamps)
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        memory_info = {}
        try:
            import psutil
            mem = psutil.virtual_memory()
            memory_info = {
                'memory_usage': f"{mem.percent}%",
                'available_memory': f"{mem.available / 1024**3:.2f} GB"
            }
        except ImportError:
            memory_info = {'memory_info': 'unavailable'}
        
        status = {
            'status': 'healthy',
            'model': MODEL_NAME,
            'model_loaded': model_loaded,
            **memory_info
        }
        if model_error:
            status['model_error'] = model_error
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Wav2Vec2 Tiny ASR API running',
        'endpoints': ['/transcribe', '/health'],
        'status': 'ready',
        'model': MODEL_NAME
    })

# =========================
# Error Handlers
# =========================
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

# =========================
# Main
# =========================
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
