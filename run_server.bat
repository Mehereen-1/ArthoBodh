@echo off
echo =======================================================
echo          Starting ArthoBodh WSD Interface Server
echo =======================================================
echo.
echo Opening http://127.0.0.1:8000 in your browser...
start http://127.0.0.1:8000
echo.
echo Loading Transformer model into memory...
.\.venv\Scripts\python.exe server.py
pause
