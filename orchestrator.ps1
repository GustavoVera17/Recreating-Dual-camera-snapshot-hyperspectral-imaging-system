param(
    [int]$Epochs = 1200,
    [int]$BatchSize = 16,
    [double]$Lr = 0.001,
    [int]$DashInterval = 10,
    [switch]$GitSync,
    [switch]$Shutdown
)

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $PSScriptRoot
$pythonPath = "C:\CODE2026\.venv\Scripts\python.exe"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   Starting Xie_Rep_2 Orchestrator (100% Replication Pipeline)        " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] Working directory : $PSScriptRoot" -ForegroundColor Gray

Set-Location -Path $PSScriptRoot

# Step 1: Run 100-Point Audit
Write-Host "[*] Executing 100-Point Pre-flight Audit Rubric..." -ForegroundColor Yellow
& "$pythonPath" audit_rubric.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Pre-flight audit failed. Aborting training to prevent invalid execution." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[+] Audit passed with 100/100 points! Starting training session..." -ForegroundColor Green

# Step 2: Optional Git Commit & Push of Initial Codebase
if ($GitSync) {
    Write-Host "[-] Syncing verified codebase to GitHub..." -ForegroundColor Cyan
    git add .
    $status = git status --porcelain
    if ($status) {
        git commit -m "feat: sync verified Xie_Rep_2 codebase before training"
        git push -u origin main
    } else {
        Write-Host "[+] Codebase already committed and synced." -ForegroundColor Green
    }
}

# Step 3: Launch Training with AMP, ETA and Dashboards
Write-Host "[*] Launching training for $Epochs epochs (Batch Size: $BatchSize, LR: $Lr)..." -ForegroundColor Yellow
$trainArgs = @(
    "train.py",
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--lr", $Lr,
    "--dash_interval", $DashInterval
)
if ($GitSync) {
    $trainArgs += "--git-sync"
}
& "$pythonPath" @trainArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Training error encountered. Exiting orchestrator." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[+] Training completed successfully!" -ForegroundColor Green

# Step 4: Run Full Evaluation across 10 Test Scenes
Write-Host "[*] Running final quantitative benchmark evaluation on 10 CAVE test scenes..." -ForegroundColor Yellow
& "$pythonPath" evaluate.py

# Step 5: Optional Final Git Push
if ($GitSync) {
    Write-Host "[-] Committing and pushing final results to GitHub..." -ForegroundColor Cyan
    git add results/ README.md
    $finalStatus = git status --porcelain
    if ($finalStatus) {
        git commit -m "Results: Completed $Epochs epochs training and 10-scene benchmark evaluation"
        git push origin main
    } else {
        Write-Host "[+] All results already committed and synced." -ForegroundColor Green
    }
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   Replication Pipeline Finished Successfully!                        " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

# Step 6: Optional Shutdown Computer
if ($Shutdown) {
    Write-Host "[*] Initiating computer shutdown in 120 seconds..." -ForegroundColor Yellow
    shutdown.exe /s /t 120 /c "Xie_Rep_2 training and benchmark evaluation finished successfully. Shutting down system."
}
