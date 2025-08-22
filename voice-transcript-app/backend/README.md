# Whisper ASR Backend

This is a Python Flask backend that integrates OpenAI's Whisper model for automatic speech recognition with word-level timestamps.

## Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend
```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Test the Health Endpoint
```bash
curl http://localhost:5000/health
```

## API Endpoints

### POST /transcribe
Transcribes audio and returns word-level timestamps.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file (WAV, MP3, etc.)

**Response:**
```json
{
  "success": true,
  "transcript": "Hello world this is a test",
  "word_timestamps": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.5
    },
    {
      "word": "world",
      "start": 0.5,
      "end": 1.0
    }
  ],
  "audio_filename": "audio_20241201_143022.wav"
}
```

## Features

- **Whisper Large v3**: State-of-the-art speech recognition
- **Word-level Timestamps**: Approximate timing for each word
- **Multiple Audio Formats**: Supports WAV, MP3, and other formats
- **SQLite Database**: Stores users and transcripts
- **CORS Enabled**: Works with frontend applications

## Notes

- The first run will download the Whisper model (~1.5GB)
- Word-level timestamps are approximated (Whisper doesn't provide exact word timing)
- For production, consider using cloud storage for audio files
- The model loads into memory on startup for faster inference

## Troubleshooting

- **CUDA Error**: If you get CUDA errors, the model will fall back to CPU (slower but works)
- **Memory Issues**: Whisper Large v3 requires ~3GB RAM. Use a smaller model if needed
- **Audio Format**: Ensure audio files are properly formatted (16kHz sample rate recommended) 