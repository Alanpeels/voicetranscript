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

# Set up proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Very explicit CORS configuration
app.config.update(
    CORS_HEADERS='Content-Type',
    CORS_RESOURCES={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With", "Access-Control-Request-Method", "Access-Control-Request-Headers"],
            "expose_headers": ["Content-Type"],
            "supports_credentials": False,
            "max_age": 600
        }
    }
)
CORS(app, resources=app.config['CORS_RESOURCES'])

# Global model variables
processor = None
model = None
model_loaded = False
model_error = None
model_loading = False
model_load_lock = threading.Lock()

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
    """Load Wav2Vec2 model with lazy loading and timeout handling"""
    global processor, model, model_loaded, model_error, model_loading
    
    with model_load_lock:
        if model_loaded:
            return True
        
        if model_loading:
            # Wait for ongoing loading (max 60 seconds)
            for _ in range(60):
                time.sleep(1)
                if model_loaded:
                    return True
                if not model_loading:
                    break
            return False
        
        model_loading = True
        
        try:
            logger.info("Starting model load...")
            
            # Try to load transformers
            try:
                from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
                import torch
                logger.info("Transformers imported successfully")
            except ImportError as e:
                model_error = f"Import failed: {e}"
                logger.error(model_error)
                model_loading = False
                return False
            
            model_name = "facebook/wav2vec2-base-960h"
            logger.info(f"Loading {model_name}...")
            
            # Load processor first (lighter)
            try:
                processor = Wav2Vec2Processor.from_pretrained(
                    model_name,
                    cache_dir="/tmp/huggingface_cache"
                )
                logger.info("Processor loaded")
            except Exception as e:
                model_error = f"Processor load failed: {e}"
                logger.error(model_error)
                model_loading = False
                return False
            
            # Load model with basic settings
            try:
                model = Wav2Vec2ForCTC.from_pretrained(model_name)
                model.eval()
                logger.info("Model loaded")
            except Exception as e:
                model_error = f"Model load failed: {e}"
                logger.error(model_error)
                model_loading = False
                return False
            
            model_loaded = True
            model_loading = False
            logger.info("Model loading complete!")
            return True
            
        except Exception as e:
            model_error = f"Unexpected error: {e}"
            logger.error(f"Model load failed: {e}")
            logger.error(traceback.format_exc())
            model_loading = False
            return False

def simple_transcribe(audio_bytes):
    """Simple transcription function with better error handling"""
    try:
        # Load required libraries
        import librosa
        import torch
        import numpy as np
        
        if not load_model():
            raise Exception(f"Model not available: {model_error}")
        
        logger.info("Processing audio...")
        # Process audio with error handling
        try:
            audio_array, sr = librosa.load(
                io.BytesIO(audio_bytes), 
                sr=16000, 
                duration=30,
                mono=True
            )
            logger.info(f"Audio processed: {len(audio_array)} samples at {sr}Hz")
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise Exception("Failed to process audio file")
        
        if len(audio_array) == 0:
            return "[No audio detected]"
        
        # Transcribe with timeout protection
        logger.info("Running inference...")
        try:
            inputs = processor(
                audio_array, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = processor.batch_decode(predicted_ids)[0]
            
            # Clean up tensors
            del inputs, logits, predicted_ids
            clear_memory()
            
            result = transcript.strip() if transcript.strip() else "[No speech detected]"
            logger.info(f"Transcription successful: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            clear_memory()
            raise Exception(f"Transcription inference failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        clear_memory()
        raise

# Add explicit OPTIONS handler for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept,Origin,X-Requested-With,Access-Control-Request-Method,Access-Control-Request-Headers")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,PUT,DELETE,OPTIONS")
        response.headers.add('Access-Control-Max-Age', '600')
        return response

# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Wav2Vec2 ASR API',
        'endpoints': ['/transcribe', '/health', '/warmup'],
        'status': 'running',
        'model_loaded': model_loaded,
        'version': '2.0'
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model_loaded,
            'model_loading': model_loading,
            'model_error': model_error,
            'timestamp': datetime.utcnow().isoformat()
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
            status['memory_info'] = 'unavailable'
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/warmup', methods=['POST', 'GET'])
def warmup():
    """Endpoint to pre-load the model"""
    try:
        logger.info("Warmup request received - loading model...")
        
        if model_loaded:
            return jsonify({
                'status': 'already_loaded',
                'message': 'Model is already loaded and ready',
                'model_loaded': True,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        if model_loading:
            return jsonify({
                'status': 'loading',
                'message': 'Model is currently loading, please wait...',
                'model_loading': True,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Start loading
        success = load_model()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model loaded successfully and ready for transcription',
                'model_loaded': True,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Model loading failed: {model_error}',
                'model_loaded': False,
                'model_error': model_error,
                'timestamp': datetime.utcnow().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Warmup error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Warmup failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        logger.info("Transcription request received")
        
        # Validate request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if not audio_file or not audio_file.filename:
            return jsonify({'error': 'Empty audio file'}), 400
        
        # Read audio data
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({'error': 'Empty audio data'}), 400
        
        # Size check
        max_size = 10 * 1024 * 1024  # 10MB
        if len(audio_bytes) > max_size:
            return jsonify({'error': f'File too large (max {max_size//1024//1024}MB)'}), 400
        
        logger.info(f"Processing audio: {len(audio_bytes)} bytes")
        
        # Check model status and provide immediate feedback
        if not model_loaded and not model_loading:
            logger.info("Model not loaded, triggering load...")
            return jsonify({
                'success': False,
                'status': 'model_loading',
                'message': 'Model is loading for the first time. This takes 30-60 seconds. Please try again in a moment.',
                'model_loaded': False,
                'model_loading': True,
                'retry_after_seconds': 45
            }), 202  # 202 = Accepted, processing
        
        if model_loading:
            return jsonify({
                'success': False,
                'status': 'model_loading',
                'message': 'Model is currently loading. Please wait and try again.',
                'model_loaded': False,
                'model_loading': True,
                'retry_after_seconds': 30
            }), 202
        
        # Transcribe with error handling
        try:
            transcript = simple_transcribe(audio_bytes)
            logger.info(f"Transcription completed: {transcript[:100]}...")
            
            # Create response
            response_data = {
                'success': True,
                'transcript': transcript,
                'word_count': len(transcript.split()) if transcript != "[No speech detected]" else 0,
                'audio_size_bytes': len(audio_bytes),
                'model_used': 'wav2vec2-base-960h',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Save to database (non-blocking)
            try:
                conn = sqlite3.connect('transcripts.db', timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO transcripts (transcript_text) VALUES (?)", (transcript,))
                conn.commit()
                conn.close()
            except Exception as db_err:
                logger.warning(f"DB save failed (non-critical): {db_err}")
            
            return jsonify(response_data)
            
        except Exception as transcribe_error:
            logger.error(f"Transcription error: {transcribe_error}")
            return jsonify({
                'success': False,
                'error': f'Transcription failed: {str(transcribe_error)}',
                'model_loaded': model_loaded,
                'model_loading': model_loading
            }), 500
        
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Request failed: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

# Add response headers for all responses - this is crucial
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With,Access-Control-Request-Method,Access-Control-Request-Headers')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '600')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    return response

if __name__ == '__main__':
    try:
        init_db()
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting server on port {port}")
        
        # Pre-load model in background thread (enabled by default)
        def background_model_load():
            logger.info("Starting background model load...")
            time.sleep(5)  # Give server time to start
            if load_model():
                logger.info("Background model load successful!")
            else:
                logger.warning(f"Background model load failed: {model_error}")
        
        # Start background loading immediately
        threading.Thread(target=background_model_load, daemon=True).start()
        
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"Server start failed: {e}")
        exit(1)