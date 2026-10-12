@echo off
setlocal
cd /d "%~dp0"

if exist "%LOCALAPPDATA%\Programs\MIVIM LoRA Studio\MIVIMLoRAStudio.exe" (
    start "MIVIM LoRA Studio" "%LOCALAPPDATA%\Programs\MIVIM LoRA Studio\MIVIMLoRAStudio.exe"
    exit /b 0
)

if not exist ".venv\Scripts\pythonw.exe" (
    echo MIVIM LoRA Studio is not installed in this folder yet.
    echo.
    echo Double-click "Setup MIVIM LoRA Studio.cmd" first.
    echo.
    pause
    exit /b 1
)

start "MIVIM LoRA Studio" ".venv\Scripts\pythonw.exe" mivim_lora_studio.py
