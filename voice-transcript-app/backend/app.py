import os
import io
import gc
import logging
import traceback
from datetime import datetime
import sqlite3

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin

# Set up proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS - be very explicit
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, 
     origins=['*'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'Accept', 'Origin', 'X-Requested-With'],
     expose_headers=['Content-Type'],
     supports_credentials=False)

# Global model variables
processor = None
model = None
model_loaded = False
model_error = None

def init_db():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect('transcripts.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

def clear_memory():
    """Clear memory"""
    gc.collect()

def load_model():
    """Load Wav2Vec2 model with better error handling"""
    global processor, model, model_loaded, model_error
    
    if model_loaded:
        return True
    
    try:
        logger.info("Starting model load...")
        
        # Try to load transformers
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            import torch
        except ImportError as e:
            model_error = f"Import failed: {e}"
            logger.error(model_error)
            return False
        
        model_name = "facebook/wav2vec2-base-960h"
        logger.info(f"Loading {model_name}...")
        
        # Load with explicit error handling
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            logger.info("Processor loaded")
        except Exception as e:
            model_error = f"Processor load failed: {e}"
            logger.error(model_error)
            return False
        
        try:
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            model.eval()
            logger.info("Model loaded")
        except Exception as e:
            model_error = f"Model load failed: {e}"
            logger.error(model_error)
            return False
        
        model_loaded = True
        logger.info("Model loading complete!")
        return True
        
    except Exception as e:
        model_error = f"Unexpected error: {e}"
        logger.error(f"Model load failed: {e}")
        logger.error(traceback.format_exc())
        return False

def simple_transcribe(audio_bytes):
    """Simple transcription function"""
    try:
        # Try to load required libraries
        import librosa
        import torch
        import numpy as np
        
        if not load_model():
            raise Exception(f"Model not available: {model_error}")
        
        # Process audio
        logger.info("Processing audio...")
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=30)
        logger.info(f"Audio processed: {len(audio_array)} samples")
        
        # Transcribe
        logger.info("Running inference...")
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = processor.batch_decode(predicted_ids)[0]
        
        # Clean up
        del inputs, logits, predicted_ids
        clear_memory()
        
        return transcript.strip() if transcript.strip() else "[No speech detected]"
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        clear_memory()
        raise

# Routes with explicit CORS
@app.route('/', methods=['GET', 'OPTIONS'])
@cross_origin()
def home():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    return jsonify({
        'message': 'Wav2Vec2 ASR API',
        'endpoints': ['/transcribe', '/health'],
        'status': 'running'
    })

@app.route('/health', methods=['GET', 'OPTIONS'])
@cross_origin()
def health():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model_loaded,
            'model_error': model_error
        }
        
        # Add memory info if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            process = psutil.Process()
            status.update({
                'memory_usage': f"{mem.percent}%",
                'available_memory': f"{mem.available / 1024**3:.2f} GB",
                'process_memory': f"{process.memory_info().rss/1024**2:.1f}MB"
            })
        except:
            pass
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/transcribe', methods=['POST', 'OPTIONS'])
@cross_origin()
def transcribe():
    # Handle preflight
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    try:
        logger.info("Transcription request received")
        
        # Check for audio file
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({'error': 'Empty audio file'}), 400
        
        # Read audio data
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({'error': 'Empty audio data'}), 400
        
        # Size check
        if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'File too large (max 10MB)'}), 400
        
        logger.info(f"Processing audio: {len(audio_bytes)} bytes")
        
        # Transcribe
        try:
            transcript = simple_transcribe(audio_bytes)
            logger.info(f"Transcription result: {transcript}")
            
            # Create response
            response_data = {
                'success': True,
                'transcript': transcript,
                'word_count': len(transcript.split()) if transcript != "[No speech detected]" else 0,
                'model_used': 'wav2vec2-base-960h'
            }
            
            # Save to database
            try:
                conn = sqlite3.connect('transcripts.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO transcripts (transcript_text) VALUES (?)", (transcript,))
                conn.commit()
                conn.close()
            except Exception as db_err:
                logger.warning(f"DB save failed: {db_err}")
            
            return jsonify(response_data)
            
        except Exception as transcribe_error:
            logger.error(f"Transcription error: {transcribe_error}")
            return jsonify({'error': f'Transcription failed: {str(transcribe_error)}'}), 500
        
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Request failed: {str(e)}'}), 500

# Add explicit CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '86400')
    return response

# Error handlers
@app.errorhandler(404)
def not_found(error):
    response = jsonify({'error': 'Not found'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    response = jsonify({'error': 'Internal server error'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 500

if __name__ == '__main__':
    try:
        init_db()
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting server on port {port}")
        
        # Test model loading on startup (optional)
        logger.info("Testing model availability...")
        if load_model():
            logger.info("Model is ready!")
        else:
            logger.warning(f"Model not ready: {model_error}")
        
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Server start failed: {e}")
        exit(1)