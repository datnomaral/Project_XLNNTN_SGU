@echo off
echo ============================================================
echo    TAO VENV MOI VOI TEN KHAC (TRANH CONFLICT)
echo ============================================================
echo.

REM Create .venv311 (fresh start)
echo [1/4] Creating .venv311 with Python 3.11...
py -3.11 -m venv .venv311
if %errorlevel% neq 0 (
    echo ERROR: Khong the tao virtual environment
    pause
    exit /b 1
)
echo OK!
echo.

REM Activate
echo [2/4] Activating .venv311...
call .venv311\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Khong the activate
    pause
    exit /b 1
)
echo OK!
echo.

REM Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip -q
echo OK!
echo.

REM Verify
echo [4/4] Verifying...
python --version
echo.

echo ============================================================
echo    THANH CONG!
echo ============================================================
echo.
echo Virtual environment: .venv311
echo Python version: 
python --version
echo.
echo Tiep theo, cai PyTorch:
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo   pip install spacy nltk matplotlib seaborn tqdm
echo.
pause
