#!/usr/bin/env python3
"""
Simple test script to verify Whisper integration
"""

import requests
import os
import time

def test_whisper_backend():
    """Test the Whisper backend endpoints"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Whisper Backend...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on port 5000")
        return False
    
    # Test 2: Test with a sample audio file (if available)
    print("\n2. Testing transcription endpoint...")
    
    # Check if we have a test audio file
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):
        print(f"   Found test audio file: {test_audio_path}")
        
        try:
            with open(test_audio_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = requests.post(f"{base_url}/transcribe", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Transcription test passed")
                    print(f"   Transcript: {result.get('transcript', 'N/A')}")
                    print(f"   Word count: {len(result.get('word_timestamps', []))}")
                else:
                    print(f"❌ Transcription test failed: {response.status_code}")
                    print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Transcription test error: {str(e)}")
    else:
        print("   ⚠️  No test audio file found. Skipping transcription test.")
        print("   Create a test_audio.wav file to test transcription.")
    
    print("\n" + "=" * 50)
    print("🎯 Backend test completed!")
    print("\nNext steps:")
    print("1. Make sure your frontend is configured to use this backend")
    print("2. Test recording and transcription through the UI")
    print("3. Check that transcripts are being saved to the database")
    
    return True

if __name__ == "__main__":
    test_whisper_backend() 