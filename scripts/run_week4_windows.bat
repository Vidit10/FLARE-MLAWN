@echo off
REM Week-4 runs (federation E8/E9 and observer Steps F-G) on Windows.
REM Run from the repository folder, inside the project's virtual environment:
REM     scripts\run_week4_windows.bat 12
REM The number is how many runs go in parallel (one core each); default 10.
REM An interrupted run can be restarted with the same command: finished runs are skipped.
setlocal
set JOBS=%1
if "%JOBS%"=="" set JOBS=10
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
if not exist results\week4 mkdir results\week4
echo Started %DATE% %TIME% with %JOBS% parallel runs
python -m experiments.week4_launch --grid all --seeds 0 1 2 --jobs %JOBS%
python -m experiments.week4_eval --part all > results\week4\eval_summary.log 2>&1
echo Finished %DATE% %TIME%
endlocal
