@echo off
setlocal
cd /d "%~dp0"
title Moto Laps launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Background -OpenBrowser
if errorlevel 1 (
  echo.
  echo Moto Laps could not start. Check data\logs\server.stderr.log
  pause
)
endlocal
