@echo off
REM AlpenEnergie — Daily Forecast Launcher
REM Double-click this file to run the full pipeline.
REM Python must be installed and on PATH, or adjust PYTHON_EXE below.

SET PYTHON_EXE=python
SET SCRIPT_DIR=%~dp0
SET MAIN_SCRIPT=%SCRIPT_DIR%CNN_LSTM\main.py

echo ============================================================
echo   AlpenEnergie Forecast Pipeline
echo   %DATE%  %TIME%
echo ============================================================

%PYTHON_EXE% "%MAIN_SCRIPT%"

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline failed. Check main_runner.log for details.
    pause
) ELSE (
    echo.
    echo [OK] Pipeline finished successfully.
    timeout /t 5
)
