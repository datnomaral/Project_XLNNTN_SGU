@echo off
echo ========================================
echo  DO AN DICH MAY ANH-PHAP
echo  Chay du an voi Python PATH day du
echo ========================================
echo.

REM Duong dan Python day du
SET PYTHON_PATH=C:\Users\OS\AppData\Local\Programs\Python\Python311\python.exe
SET PIP_PATH=C:\Users\OS\AppData\Local\Programs\Python\Python311\Scripts\pip.exe

echo [1/6] Kiem tra Python...
%PYTHON_PATH% --version
if errorlevel 1 (
    echo ERROR: Khong tim thay Python!
    pause
    exit /b 1
)
echo    Python OK!

REM Tao virtual environment
echo.
echo [2/6] Tao virtual environment...
if not exist venv (
    %PYTHON_PATH% -m venv venv
    echo    Virtual environment da tao!
) else (
    echo    Virtual environment da ton tai!
)

REM Kich hoat virtual environment
echo.
echo [3/6] Kich hoat virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/6] Nang cap pip...
python -m pip install --upgrade pip --quiet

REM Cai dat dependencies
echo.
echo [5/6] Cai dat cac thu vien can thiet...
echo    (Co the mat 5-10 phut, vui long doi...)
pip install torch torchtext numpy pandas spacy nltk matplotlib seaborn jupyter notebook tqdm --quiet

REM Download spaCy models
echo.
echo [6/6] Download spaCy language models...
python -m spacy download en_core_web_sm --quiet
python -m spacy download fr_core_news_sm --quiet

REM Mo Jupyter Notebook
echo.
echo ========================================
echo  Setup hoan tat!
echo  Dang mo Jupyter Notebook...
echo ========================================
echo.
echo Trinh duyet se mo trong giay lat...
echo Trong notebook:
echo   - Chay: Cell ^> Run All
echo   - Hoac: Shift + Enter tung cell
echo.
echo Thoi gian training: 30-60 phut (CPU)
echo ========================================
echo.

jupyter notebook main.ipynb

echo.
echo ========================================
echo  KET THUC!
echo ========================================
pause
