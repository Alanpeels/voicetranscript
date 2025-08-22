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

app = Flask(__name__)

# Configure CORS properly
CORS(app, 
     origins=["*"],  # Allow all origins for now
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

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Global variables for model (lazy loading)
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        print("Loading Whisper model...")
        # Use smaller model to save memory
        model_name = "openai/whisper-small"  # Changed from whisper-large-v3
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Move to CPU and set to eval mode to save memory
        model.eval()
        
        print(f"Whisper model {model_name} loaded successfully!")

def clear_memory():
    """Clear memory after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def init_db():
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

def process_audio_directly(audio_bytes, filename):
    try:
        # Limit audio length to save memory
        audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=300)  # Max 5 minutes
        print(f"Successfully loaded audio directly: {len(audio_array)} samples, {sample_rate}Hz")
        return audio_array, sample_rate
    except Exception as e:
        print(f"Direct loading failed: {str(e)}")
        
        try:
            audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, duration=300)  # Max 5 minutes
            print(f"Loaded with original sample rate: {sample_rate}Hz")
            
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
                print(f"Resampled to 16kHz")
            
            return audio_array, sample_rate
        except Exception as e2:
            print(f"Fallback loading also failed: {str(e2)}")
            raise Exception(f"Could not process audio file {filename}. Supported formats: WAV, MP3, OGG, FLAC, M4A")

def get_word_timestamps(audio_array, sample_rate):
    load_model()  # Lazy load the model
    
    # Process in smaller chunks if needed
    max_length = 16000 * 30  # 30 seconds max at a time
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
        print(f"Truncated audio to 30 seconds to save memory")
    
    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    
    with torch.no_grad():  # Don't compute gradients to save memory
        predicted_ids = model.generate(
            inputs["input_features"],
            return_timestamps=True,
            max_length=448,  # Limit output length
            num_beams=1,     # Use greedy decoding to save memory
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

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():        
    try:
        print(f"Received request with method: {request.method}")
        print(f"Request headers: {dict(request.headers)}")
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Check file size (limit to 10MB)
        if len(audio_bytes) > 10 * 1024 * 1024:
            return jsonify({'error': 'Audio file too large. Maximum size is 10MB.'}), 400
        
        filename = audio_file.filename or 'audio'
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'webm'
        
        print(f"Processing audio file: {filename} (format: {file_extension})")
        print(f"Audio file size: {len(audio_bytes)} bytes")
        
        audio_array, sample_rate = process_audio_directly(audio_bytes, filename)
        
        print(f"Audio loaded: {len(audio_array)} samples, {sample_rate}Hz sample rate")
        
        word_timestamps = get_word_timestamps(audio_array, sample_rate)
        
        transcript_text = " ".join([word['word'] for word in word_timestamps])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"
        
        print(f"Transcription completed: {len(word_timestamps)} words")
        
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
        print(f"Error in transcription: {str(e)}")
        clear_memory()  # Clear memory even on error
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'whisper-small'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Voice Transcript API is running', 'endpoints': ['/transcribe', '/health']})

if __name__ == '__main__':
    init_db()
    print("Starting Flask server...")
    # Use the port that Render provides via environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)