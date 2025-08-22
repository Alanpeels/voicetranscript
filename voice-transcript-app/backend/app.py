from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import requests
from datetime import datetime
import sqlite3
import io

app = Flask(__name__)

# CORS configuration
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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

def transcribe_with_openai(audio_bytes, filename):
    """
    Use OpenAI's Whisper API for transcription
    You'll need to set OPENAI_API_KEY environment variable
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name
    
    try:
        # Call OpenAI Whisper API
        with open(temp_file_path, 'rb') as audio_file:
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                },
                files={
                    'file': (filename, audio_file, 'audio/wav'),
                },
                data={
                    'model': 'whisper-1',
                    'response_format': 'verbose_json',
                    'timestamp_granularities': ['word']
                }
            )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Extract word timestamps if available
        word_timestamps = []
        if 'words' in result:
            word_timestamps = [
                {
                    "word": word['word'],
                    "start": word['start'],
                    "end": word['end']
                }
                for word in result['words']
            ]
        else:
            # Fallback: create approximate timestamps
            words = result['text'].split()
            duration = result.get('duration', len(words))  # Approximate
            words_per_second = len(words) / duration if duration > 0 else 0
            
            for i, word in enumerate(words):
                start_time = i / words_per_second if words_per_second > 0 else 0
                end_time = (i + 1) / words_per_second if words_per_second > 0 else 0
                
                word_timestamps.append({
                    "word": word,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3)
                })
        
        return result['text'], word_timestamps
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def transcribe_with_local_model(audio_bytes, filename):
    """
    Fallback: Simple mock transcription for development/testing
    Replace this with your preferred local transcription method
    """
    # This is a mock implementation - replace with actual transcription
    mock_text = "This is a mock transcription. Please configure OpenAI API key for real transcription."
    words = mock_text.split()
    
    word_timestamps = []
    for i, word in enumerate(words):
        word_timestamps.append({
            "word": word,
            "start": round(i * 0.5, 3),  # Mock timing
            "end": round((i + 1) * 0.5, 3)
        })
    
    return mock_text, word_timestamps

@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Check file size (limit to 25MB - OpenAI's limit)
        if len(audio_bytes) > 25 * 1024 * 1024:
            return jsonify({'error': 'Audio file too large. Maximum size is 25MB.'}), 400
        
        filename = audio_file.filename or 'audio.wav'
        
        print(f"Processing audio file: {filename}")
        print(f"Audio file size: {len(audio_bytes)} bytes")
        
        try:
            # Try OpenAI API first
            transcript_text, word_timestamps = transcribe_with_openai(audio_bytes, filename)
            print("Used OpenAI Whisper API")
        except Exception as openai_error:
            print(f"OpenAI API failed: {str(openai_error)}")
            # Fallback to local/mock transcription
            transcript_text, word_timestamps = transcribe_with_local_model(audio_bytes, filename)
            print("Used fallback transcription")
        
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
    openai_configured = bool(os.environ.get('OPENAI_API_KEY'))
    return jsonify({
        'status': 'healthy',
        'openai_api_configured': openai_configured,
        'transcription_method': 'OpenAI Whisper API' if openai_configured else 'Local/Mock'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Voice Transcript API is running',
        'endpoints': ['/transcribe', '/health'],
        'note': 'Configure OPENAI_API_KEY environment variable for best results'
    })

if __name__ == '__main__':
    init_db()
    print("Starting lightweight Flask server...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)