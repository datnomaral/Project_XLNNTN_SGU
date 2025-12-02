@echo off
echo ============================================================
echo    CAI DAT PYTORCH VOI CUDA CHO RTX 4060
echo ============================================================
echo.

REM Check Python version
echo [0/6] Checking Python version...
python --version
python -c "import sys; ver=sys.version_info; exit(0 if ver.major==3 and ver.minor in [11,12] else 1)"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python version khong hop le!
    echo PyTorch yeu cau Python 3.11 hoac 3.12
    echo Ban dang dung Python 3.13
    echo.
    echo Vui long:
    echo   1. Cai Python 3.11: https://www.python.org/downloads/release/python-3115/
    echo   2. Chay script: setup_python311.bat
    echo   3. Chay lai script nay
    echo.
    pause
    exit /b 1
)
echo OK! Python version hop le.
echo.

REM Activate virtual environment
echo [1/6] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Khong the activate .venv
    pause
    exit /b 1
)
echo OK!
echo.

REM Uninstall old PyTorch
echo [2/6] Uninstalling old PyTorch...
pip uninstall -y torch torchvision torchaudio torchtext
echo OK!
echo.

REM Install PyTorch with CUDA 12.1
echo [3/6] Installing PyTorch with CUDA 12.1...
echo (Qua trinh nay co the mat 2-5 phut...)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Khong the cai dat PyTorch
    pause
    exit /b 1
)
echo OK!
echo.

REM Reinstall dependencies
echo [4/6] Reinstalling dependencies...
pip install spacy nltk matplotlib seaborn tqdm
echo OK!
echo.

REM Download spaCy models
echo [5/6] Downloading spaCy models...
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
echo OK!
echo.

REM Verify CUDA
echo [6/6] Verifying CUDA support...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

if %errorlevel% equ 0 (
    echo ============================================================
    echo    THANH CONG! GPU da san sang!
    echo ============================================================
    echo.
    echo Tiep theo:
    echo   1. Restart Jupyter Notebook (Ctrl+C roi chay lai)
    echo   2. Restart kernel trong notebook
    echo   3. Chay lai training - se nhanh gap 10-20 lan!
    echo.
) else (
    echo ============================================================
    echo    CO LOI XAY RA!
    echo ============================================================
    echo Vui long kiem tra ket qua phia tren
)

pause
