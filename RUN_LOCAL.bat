@echo off
echo ========================================
echo  DO AN DICH MAY ANH-PHAP
echo  Setup va chay du an
echo ========================================
echo.

REM Kiem tra Python
echo [1/6] Kiem tra Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python chua duoc cai dat!
    echo.
    echo Vui long:
    echo 1. Khoi dong lai PowerShell/Terminal
    echo 2. Hoac tai Python tai: https://www.python.org/downloads/
    echo 3. Hoac dung Google Colab (xem QUICK_START.md)
    pause
    exit /b 1
)
echo    Python da duoc cai dat!

REM Tao virtual environment
echo.
echo [2/6] Tao virtual environment...
if not exist venv (
    python -m venv venv
    echo    Virtual environment da tao!
) else (
    echo    Virtual environment da ton tai!
)

REM Kich hoat virtual environment
echo.
echo [3/6] Kich hoat virtual environment...
call venv\Scripts\activate.bat

REM Cai dat dependencies
echo.
echo [4/6] Cai dat cac thu vien can thiet...
echo    (Co the mat 5-10 phut...)
pip install --upgrade pip
pip install -r requirements.txt

REM Download spaCy models
echo.
echo [5/6] Download spaCy language models...
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

REM Mo Jupyter Notebook
echo.
echo [6/6] Mo Jupyter Notebook...
echo.
echo ========================================
echo  Setup hoan tat!
echo  Trinh duyet se mo main.ipynb
echo  Chay: Runtime -^> Run all
echo ========================================
echo.
jupyter notebook main.ipynb

pause
