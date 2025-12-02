@echo off
echo ============================================================
echo    CAI DAT TAT CA DEPENDENCIES
echo ============================================================
echo.

REM Activate venv
call .venv\Scripts\activate.bat

echo [1/3] Installing Jupyter...
pip install jupyter notebook ipykernel -q
echo OK!
echo.

echo [2/3] Installing project dependencies...
pip install pandas numpy -q
echo OK!
echo.

echo [3/3] Verifying...
echo.
echo Python version:
python --version
echo.
echo PyTorch + CUDA:
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo.
echo Jupyter:
jupyter --version
echo.

echo ============================================================
echo    THANH CONG!
echo ============================================================
echo.
echo Tiep theo:
echo   jupyter notebook main.ipynb
echo.
pause
