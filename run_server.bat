@echo off
echo Starting ArthoBodh (loading BanglaBERT can take ~20 seconds)...
start "" cmd /c "timeout /t 20 >nul & start http://127.0.0.1:8000"
.\venv\Scripts\python.exe server.py
pause
