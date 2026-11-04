@echo off
rem Starts scripts\run_week3_windows.ps1 (week-3 runs for seeds 1 and 2). Close this window to stop.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_week3_windows.ps1" %*
pause
