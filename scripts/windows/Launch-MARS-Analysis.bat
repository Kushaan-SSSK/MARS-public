@echo off
setlocal

cd /d "%~dp0"
cd /d "..\.."

set "CONFIG=%~1"
if "%CONFIG%"=="" if not "%MARS_ANALYSIS_CONFIG%"=="" set "CONFIG=%MARS_ANALYSIS_CONFIG%"
if "%CONFIG%"=="" if exist "tools\mars_analysis\analysis_config.local.json" set "CONFIG=tools\mars_analysis\analysis_config.local.json"
if "%CONFIG%"=="" set "CONFIG=tools\mars_analysis\analysis_config.example.json"

if not "%MARS_ANALYSIS_PYTHON%"=="" (
    set "PY=%MARS_ANALYSIS_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "..\..\AccuSleePy\.venv_cuda\Scripts\python.exe" (
    set "PY=..\..\AccuSleePy\.venv_cuda\Scripts\python.exe"
) else if exist "..\..\AccuSleePy\.venv\Scripts\python.exe" (
    set "PY=..\..\AccuSleePy\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" -m tools.mars_analysis gui "%CONFIG%"
if errorlevel 1 (
    echo.
    echo MARS Analysis GUI failed to launch.
    echo Python: "%PY%"
    echo Config: "%CONFIG%"
    echo.
    pause
)
