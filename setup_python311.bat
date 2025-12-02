@echo off
echo ============================================================
echo    SETUP LAI VIRTUAL ENVIRONMENT VOI PYTHON 3.11
echo ============================================================
echo.

REM Check Python 3.11
echo [1/6] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 chua duoc cai dat!
    echo.
    echo Vui long:
    echo   1. Tai Python 3.11.5 tai: https://www.python.org/downloads/release/python-3115/
    echo   2. Chon "Windows installer (64-bit)"
    echo   3. CHECK "Add Python to PATH" khi cai dat
    echo   4. Restart terminal va chay lai script nay
    echo.
    pause
    exit /b 1
)
py -3.11 --version
echo OK!
echo.

REM Deactivate current venv
echo [2/6] Deactivating current virtual environment...
call deactivate 2>nul
echo OK!
echo.

REM Remove old .venv
echo [3/6] Removing old .venv (Python 3.13)...
if exist .venv (
    rmdir /s /q .venv
    echo Removed old .venv
) else (
    echo No old .venv found
)
echo OK!
echo.

REM Create new .venv with Python 3.11
echo [4/6] Creating new .venv with Python 3.11...
py -3.11 -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Khong the tao virtual environment
    pause
    exit /b 1
)
echo OK!
echo.

REM Activate new venv
echo [5/6] Activating new virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Khong the activate .venv
    pause
    exit /b 1
)
echo OK!
echo.

REM Upgrade pip
echo [6/6] Upgrading pip...
python -m pip install --upgrade pip
echo OK!
echo.

echo ============================================================
echo    THANH CONG! Virtual environment da san sang!
echo ============================================================
echo.
echo Tiep theo, chay lenh:
echo   .\install_pytorch_gpu.bat
echo.
pause
