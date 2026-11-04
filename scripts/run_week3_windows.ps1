# Week-3 runs for further training seeds on a Windows machine (Python simulator).
#
# 1. Finds a Python with the required packages (or creates .venv and installs them).
# 2. Measures the current CPU load and uses only the cores that are free,
#    keeping $Reserve logical cores unused. All processes run at below-normal priority.
# 3. Fine-tunes the six week-3 runs for the given seeds, then runs the E6
#    evaluation for each seed into results\week3\seed<N>\. Windows has no fork(), so the
#    evaluation is split into shards that run as separate processes and are merged afterwards
#    (every episode has its own seed, so the result does not depend on the split).
# The machine is kept awake while the script runs. Log: results\week3\windows_run_<time>.log
#
# Usage: double-click scripts\run_week3_windows.bat, or
#   powershell -ExecutionPolicy Bypass -File scripts\run_week3_windows.ps1 [-Seeds 1,2] [-Reserve 4] [-MaxJobs 12]
param([int[]]$Seeds = @(1, 2), [int]$Reserve = 4, [int]$MaxJobs = 12, [int]$Episodes = 8)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "results\week3" | Out-Null
$Log = "results\week3\windows_run_$(Get-Date -Format yyyyMMdd_HHmmss).log"
Start-Transcript -Path $Log | Out-Null
(Get-Process -Id $PID).PriorityClass = "BelowNormal"     # child processes inherit it

# keep the machine awake while this script runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
Add-Type -Namespace W -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

function Py { & $script:py @args 2>&1 | ForEach-Object { "$_" } | Out-Host; return $LASTEXITCODE }

try {
    # ---------------------------------------------------------------- Python
    $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:PYTHONUNBUFFERED = "1"
    $check = "import numpy, pandas, matplotlib, torch, pytest; print('torch', torch.__version__)"
    $script:py = $null
    # ask each candidate for its own path: "python" can be a Store alias whose
    # Get-Command .Source is empty, which made the first version of this script stop
    foreach ($cand in @(".venv\Scripts\python.exe", "python", "py")) {
        if (-not (Get-Command $cand -ErrorAction SilentlyContinue)) { continue }
        $exe = (& $cand -c "$check; import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
        if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe)) { $script:py = "$exe".Trim(); break }
    }
    if (-not $script:py) {
        $base = $null
        foreach ($cand in @("py", "python")) {
            if (-not (Get-Command $cand -ErrorAction SilentlyContinue)) { continue }
            & $cand -c "import sys; assert sys.version_info >= (3, 10)" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $base = $cand; break }
        }
        if (-not $base) { throw "No Python 3.10+ found. Install Python from python.org and run this script again." }
        Write-Host "Creating .venv and installing the requirements (CPU build of torch)..."
        & $base -m venv .venv
        if (-not (Test-Path ".venv\Scripts\python.exe")) {
            throw "Could not create .venv with '$base'. Create it by hand (python -m venv .venv) and run this script again."
        }
        $script:py = (Resolve-Path ".venv\Scripts\python.exe").Path
        Py -m pip install --upgrade pip | Out-Null
        if ((Py -m pip install torch --index-url https://download.pytorch.org/whl/cpu) -ne 0) { throw "torch installation failed." }
        if ((Py -m pip install -r requirements.txt) -ne 0) { throw "Installing requirements.txt failed." }
    }
    Py -c $check | Out-Null
    Write-Host "Python: $script:py"
    if ((Py -m pytest -q --color=no tests) -ne 0) { throw "Tests failed; not starting the runs." }

    # ---------------------------------------------------------------- cores
    $n = [Environment]::ProcessorCount
    $samples = @()
    for ($i = 0; $i -lt 10; $i++) {
        $samples += (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        Start-Sleep -Seconds 2
    }
    $load = ($samples | Measure-Object -Average).Average
    $busy = [math]::Ceiling($n * $load / 100)
    $jobs = [int][math]::Max(1, [math]::Min($MaxJobs, $n - $busy - $Reserve))
    Write-Host ("Logical cores {0}, average load over 20 s {1:N0}% (about {2} cores busy), reserve {3} -> {4} parallel jobs" -f $n, $load, $busy, $Reserve, $jobs)

    # ---------------------------------------------------------------- training
    $t0 = Get-Date
    if ((Py -m experiments.week3_train --runs all --seeds $Seeds --jobs $jobs) -ne 0) { throw "Training failed (see results\week3\train_*.log)." }
    Write-Host ("Training finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)

    # ---------------------------------------------------------------- evaluation
    foreach ($s in $Seeds) {
        $t1 = Get-Date
        $dir = "results\week3\seed$s"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $procs = @()
        for ($k = 0; $k -lt $jobs; $k++) {
            $p = Start-Process -FilePath $script:py -NoNewWindow -PassThru `
                -ArgumentList @("-m", "experiments.week3_eval", "--part", "e6", "--episodes", "$Episodes", "--seed", "$s",
                                "--workers", "1", "--shard", "$k", "--n-shards", "$jobs", "--save-dir", $dir) `
                -RedirectStandardOutput "$dir\eval_shard$k.log" -RedirectStandardError "$dir\eval_shard$k.err"
            $null = $p.Handle          # keeps the exit code readable after the process ends
            $procs += $p
        }
        $procs | Wait-Process
        foreach ($p in $procs) { $p.Refresh(); if ($p.ExitCode -ne 0) { throw "An evaluation shard of seed $s failed (see $dir\eval_shard*.err)." } }
        if ((Py -m experiments.week3_eval --part merge --n-shards $jobs --seed $s --save-dir $dir) -ne 0) { throw "Merging seed $s failed." }
        Write-Host ("E6 for seed {0} finished in {1:N0} min" -f $s, ((Get-Date) - $t1).TotalMinutes)
    }
    Write-Host "ALL DONE"
}
catch { Write-Host "STOPPED: $_" }
finally {
    [W.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
    Stop-Transcript | Out-Null
}
