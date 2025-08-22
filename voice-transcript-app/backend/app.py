from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
import io
import base64
import json
from datetime import datetime
import sqlite3
import os
import tempfile
import gc
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS properly
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
     supports_credentials=False)

# Handle preflight requests globally
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Global variables for model (lazy loading)
processor = None
model = None
model_loaded = False
model_error = None

def load_model():
    global processor, model, model_loaded, model_error
    if model_loaded:
        return True
        
    try:
        logger.info("Loading Whisper model...")
        model_name = "openai/whisper-small"
        
        # Check available memory
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Available memory: {memory.available / 1024**3:.2f} GB")
        
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        
        model_loaded = True
        logger.info(f"Whisper model {model_name} loaded successfully!")
        return True
        
    except Exception as e:
        model_error = str(e)
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        return False

def clear_memory():
    """Clear memory after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def init_db():
    try:
        conn = sqlite3.connect('transcripts.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                audio_filename TEXT,
                transcript_text TEXT,
                word_timestamps TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def process_audio_directly(audio_bytes, filename):
    try:
        # Limit audio length to save memory
        audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=60)  # Reduced to 1 minute
        logger.info(f"Successfully loaded audio directly: {len(audio_array)} samples, {sample_rate}Hz")
        return audio_array, sample_rate
    except Exception as e:
        logger.error(f"Direct loading failed: {str(e)}")
        
        try:
            audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, duration=60)
            logger.info(f"Loaded with original sample rate: {sample_rate}Hz")
            
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
                logger.info(f"Resampled to 16kHz")
            
            return audio_array, sample_rate
        except Exception as e2:
            logger.error(f"Fallback loading also failed: {str(e2)}")
            raise Exception(f"Could not process audio file {filename}. Supported formats: WAV, MP3, OGG, FLAC, M4A")

def get_word_timestamps(audio_array, sample_rate):
    if not load_model():
        raise Exception(f"Model loading failed: {model_error}")
    
    # Process in smaller chunks if needed
    max_length = 16000 * 30  # 30 seconds max
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
        logger.info(f"Truncated audio to 30 seconds")
    
    try:
        inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                return_timestamps=True,
                max_length=224,  # Reduced further
                num_beams=1,
            )
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Approximate word timestamps
        words = transcription.split()
        word_timestamps = []
        
        duration = len(audio_array) / sample_rate
        words_per_second = len(words) / duration if duration > 0 else 0
        
        for i, word in enumerate(words):
            start_time = i / words_per_second if words_per_second > 0 else 0
            end_time = (i + 1) / words_per_second if words_per_second > 0 else 0
            
            word_timestamps.append({
                "word": word,
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
        
        # Clear memory after processing
        del inputs, predicted_ids
        clear_memory()
        
        return word_timestamps
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        clear_memory()
        raise

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():        
    try:
        logger.info(f"Received transcription request")
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Check file size (reduced limit for free tier)
        if len(audio_bytes) > 5 * 1024 * 1024:  # 5MB limit
            return jsonify({'error': 'Audio file too large. Maximum size is 5MB.'}), 400
        
        filename = audio_file.filename or 'audio'
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'webm'
        
        logger.info(f"Processing audio file: {filename} (format: {file_extension})")
        logger.info(f"Audio file size: {len(audio_bytes)} bytes")
        
        audio_array, sample_rate = process_audio_directly(audio_bytes, filename)
        
        logger.info(f"Audio loaded: {len(audio_array)} samples, {sample_rate}Hz sample rate")
        
        word_timestamps = get_word_timestamps(audio_array, sample_rate)
        
        transcript_text = " ".join([word['word'] for word in word_timestamps])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"
        
        logger.info(f"Transcription completed: {len(word_timestamps)} words")
        
        # Clear variables to free memory
        del audio_bytes, audio_array
        clear_memory()
        
        return jsonify({
            'success': True,
            'transcript': transcript_text,
            'word_timestamps': word_timestamps,
            'audio_filename': audio_filename
        })
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        logger.error(traceback.format_exc())
        clear_memory()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        
        status = {
            'status': 'healthy',
            'model': 'whisper-small',
            'model_loaded': model_loaded,
            'memory_usage': f"{memory.percent}%",
            'available_memory': f"{memory.available / 1024**3:.2f} GB"
        }
        
        if model_error:
            status['model_error'] = model_error
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    try:
        return jsonify({
            'message': 'Voice Transcript API is running', 
            'endpoints': ['/transcribe', '/health'],
            'status': 'ready'
        })
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return jsonify({'error': 'Service temporarily unavailable'}), 500

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    try:
        init_db()
        logger.info("Starting Flask server...")
        port = int(os.environ.get('PORT', 5000))
        
        # Don't preload the model on startup to avoid timeout
        logger.info(f"Server starting on port {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)