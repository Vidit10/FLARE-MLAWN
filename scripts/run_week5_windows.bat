@echo off
REM Week-5 runs on Windows: federation size x users (fl_scale, fl_learn) and the
REM single-cell comparison of every controller under every attack (compare).
REM Run from the repository folder, inside the project's virtual environment:
REM     scripts\run_week5_windows.bat 12
REM The number is how many runs go in parallel (one core each); default 4.
REM An interrupted run can be restarted with the same command: finished runs are skipped.
setlocal
set JOBS=%1
if "%JOBS%"=="" set JOBS=4
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
if not exist results\week5 mkdir results\week5
echo Started %DATE% %TIME% with %JOBS% parallel runs
python -m experiments.week5_launch --grid all --seeds 0 1 2 --jobs %JOBS%
python -m experiments.week5_eval > results\week5\eval_summary.log 2>&1
echo Finished %DATE% %TIME%
endlocal
