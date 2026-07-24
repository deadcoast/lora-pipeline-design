@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.11 or newer from https://www.python.org/downloads/windows/
    echo Choose "Add Python to PATH" during installation, then run this setup again.
    pause
    exit /b 1
)

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo Python 3.11 or newer is required.
    echo Install a current 64-bit Python release, then run this setup again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the private MIVIM LoRA Studio environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo The existing .venv uses an unsupported Python version.
    echo Rename or remove the .venv folder, then run this setup again.
    pause
    exit /b 1
)

echo Installing MIVIM LoRA Studio...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Setup complete. Opening MIVIM LoRA Studio...
call "Start MIVIM LoRA Studio.cmd"
exit /b 0

:failed
echo.
echo Setup did not finish. Check your internet connection and the message above.
pause
exit /b 1
