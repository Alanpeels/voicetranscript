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

app = Flask(__name__)
CORS(app, origins=[
    "https://voicetranscript-gm1e.vercel.app",
    "https://voicetranscript.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000"
], supports_credentials=True)

print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
print("Whisper model loaded successfully!")

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
        audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        print(f"Successfully loaded audio directly: {len(audio_array)} samples, {sample_rate}Hz")
        return audio_array, sample_rate
    except Exception as e:
        print(f"Direct loading failed: {str(e)}")
        
        try:
            audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
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
    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    
    predicted_ids = model.generate(
        inputs["input_features"],
        return_timestamps=True,
        output_scores=True
    )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Approximate word timestamps (Whisper doesn't provide exact word-level timestamps)
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
    
    return word_timestamps

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
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
        
        return jsonify({
            'success': True,
            'transcript': transcript_text,
            'word_timestamps': word_timestamps,
            'audio_filename': audio_filename
        })
        
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'whisper-large-v3'})

if __name__ == '__main__':
    init_db()
    print("Starting Flask server...")
    # Use the port that Render provides via environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 