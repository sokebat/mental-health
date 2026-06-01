@echo off
cd /d "%~dp0"

if not exist venv\ (
    echo Virtual environment not found. Running setup first...
    call setup.bat
)

echo.
echo Starting Mental Health Classifier...
call venv\Scripts\activate.bat
streamlit run app.py
