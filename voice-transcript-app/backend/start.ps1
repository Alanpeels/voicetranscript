Write-Host "Starting Whisper ASR Backend..." -ForegroundColor Green
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Yellow
python app.py 