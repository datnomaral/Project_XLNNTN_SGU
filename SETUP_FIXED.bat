@echo off
REM Machine Translation Setup Script - FIXED VERSION
REM No need to add Python to PATH - We use full path

echo ========================================
echo    MACHINE TRANSLATION PROJECT SETUP
echo    (Fixed - No torchtext)
echo ========================================
echo.

REM Set Python path
set PYTHON_PATH=C:\Users\OS\AppData\Local\Programs\Python\Python311\python.exe

echo [1/6] Checking Python...
"%PYTHON_PATH%" --version
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo.

echo [2/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    "%PYTHON_PATH%" -m venv venv
)
echo.

echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [5/6] Installing dependencies (WITHOUT torchtext)...
pip install -r requirements_fixed.txt
echo.

echo [6/6] Downloading spaCy models...
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
echo.

echo ========================================
echo    SETUP COMPLETE!
echo ========================================
echo.
echo Starting Jupyter Notebook...
echo.
jupyter notebook main.ipynb

pause
