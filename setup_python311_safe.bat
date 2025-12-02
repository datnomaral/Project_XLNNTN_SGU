@echo off
echo ============================================================
echo    SETUP LAI VIRTUAL ENVIRONMENT VOI PYTHON 3.11 (SAFE MODE)
echo ============================================================
echo.

REM Check Python 3.11
echo [1/5] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 chua duoc cai dat!
    pause
    exit /b 1
)
py -3.11 --version
echo OK!
echo.

REM Deactivate current venv
echo [2/5] Deactivating current virtual environment...
call deactivate 2>nul
echo OK!
echo.

REM Rename old .venv instead of deleting (avoid permission issues)
echo [3/5] Renaming old .venv to .venv_old...
if exist .venv (
    if exist .venv_old (
        echo Removing previous .venv_old...
        rmdir /s /q .venv_old 2>nul
    )
    echo Renaming .venv to .venv_old...
    rename .venv .venv_old
    if %errorlevel% neq 0 (
        echo WARNING: Khong the rename .venv
        echo Vui long:
        echo   1. Dong het Jupyter Notebook
        echo   2. Dong het terminal/IDE dang dung Python
        echo   3. Chay lai script nay
        pause
        exit /b 1
    )
    echo Renamed successfully
) else (
    echo No .venv found
)
echo OK!
echo.

REM Create new .venv with Python 3.11
echo [4/5] Creating new .venv with Python 3.11...
py -3.11 -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Khong the tao virtual environment
    pause
    exit /b 1
)
echo OK!
echo.

REM Activate new venv
echo [5/5] Activating new virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Khong the activate .venv
    pause
    exit /b 1
)
echo OK!
echo.

REM Upgrade pip
echo [6/5] Upgrading pip...
python -m pip install --upgrade pip -q
echo OK!
echo.

echo ============================================================
echo    THANH CONG! Virtual environment da san sang!
echo ============================================================
echo.
echo Python version:
python --version
echo.
echo Luu y: .venv_old van con trong thu muc
echo Ban co the xoa no sau khi restart may:
echo   rmdir /s /q .venv_old
echo.
echo Tiep theo, chay lenh:
echo   .\install_pytorch_gpu.bat
echo.
pause
