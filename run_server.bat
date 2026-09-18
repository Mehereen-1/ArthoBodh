@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Starting ArthoBodh (loading BanglaBERT can take ~20 seconds)...
start "" cmd /c "timeout /t 20 >nul & start http://127.0.0.1:8000"
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe server.py
) else if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe server.py
) else (
    python server.py
)
pause
