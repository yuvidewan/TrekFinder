@echo off
rem TrekFinder engine demo: double-click to start. Opens http://127.0.0.1:8765 in your browser.
cd /d "%~dp0lab"
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating Python environment...
  python -m venv .venv || (echo Python 3.10+ is required. & pause & exit /b 1)
)
echo Checking packages...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if not exist "corpus\manifest.json" (echo Corpus manifest missing. & pause & exit /b 1)
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" demo\server.py
pause
