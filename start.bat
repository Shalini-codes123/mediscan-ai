@echo off
REM MediScan AI — Windows startup script
REM Usage: double-click or run from terminal

echo.
echo ==========================================
echo       MediScan AI -- Starting Up
echo ==========================================
echo.

REM Check if venv exists
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating venv and installing backend deps...
call venv\Scripts\activate.bat
pip install -r backend\requirements.txt -q

REM Check models
set MODEL_COUNT=0
for %%f in (backend\models\*.pkl) do set /a MODEL_COUNT+=1

if %MODEL_COUNT% LSS 6 (
    echo.
    echo WARNING: Only %MODEL_COUNT%/6 models found.
    echo Run first: python scripts\download_and_train.py
    echo Requires Kaggle API key at %%USERPROFILE%%\.kaggle\kaggle.json
    echo.
    pause
)

REM Install frontend deps
if not exist frontend\node_modules (
    echo Installing frontend dependencies...
    cd frontend && npm install && cd ..
)

REM Start Flask in new window
echo Starting Flask backend on http://localhost:5000
start "MediScan Backend" cmd /k "cd backend && python app.py"

REM Wait a moment then start Vite
timeout /t 2 >nul
echo Starting React frontend on http://localhost:3000
start "MediScan Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo   MediScan AI is running!
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:5000
echo   Close the terminal windows to stop.
echo ==========================================
echo.
pause