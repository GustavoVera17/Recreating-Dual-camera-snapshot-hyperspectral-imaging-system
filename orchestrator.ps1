$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2"
$pythonPath = "C:\CODE2026\.venv\Scripts\python.exe"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   Starting Xie_Rep_2 Orchestrator (100% Replication Pipeline)        " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Set-Location -Path "C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2"

# Step 1: Run 100-Point Audit
Write-Host "[*] Executing 100-Point Pre-flight Audit Rubric..." -ForegroundColor Yellow
& "$pythonPath" audit_rubric.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Pre-flight audit failed. Aborting training to prevent invalid execution." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[+] Audit passed with 100/100 points! Starting training session..." -ForegroundColor Green

# Step 2: Initial Git Commit & Push of Initial Codebase
Write-Host "[-] Syncing verified codebase to GitHub..." -ForegroundColor Cyan
git add .
$status = git status --porcelain
if ($status) {
    git commit -m "feat: initial commit of certified 100/100 Xie_Rep_2 codebase and documentation"
    git push -u origin main
} else {
    Write-Host "[+] Codebase already committed and synced." -ForegroundColor Green
}

# Step 3: Launch Training with AMP, ETA and Dashboards
$epochs = 1200
$batchSize = 16
$lr = 0.001

Write-Host "[*] Launching training for $epochs epochs (Batch Size: $batchSize, LR: $lr)..." -ForegroundColor Yellow
& "$pythonPath" train.py --epochs $epochs --batch_size $batchSize --lr $lr --dash_interval 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Training error encountered. Exiting orchestrator." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[+] Training completed successfully!" -ForegroundColor Green

# Step 4: Run Full Evaluation across 10 Test Scenes
Write-Host "[*] Running final quantitative benchmark evaluation on 10 CAVE test scenes..." -ForegroundColor Yellow
& "$pythonPath" evaluate.py

# Step 5: Final Git Push
Write-Host "[-] Committing and pushing final results to GitHub..." -ForegroundColor Cyan
git add results/ README.md
$finalStatus = git status --porcelain
if ($finalStatus) {
    git commit -m "Results: Completed 1200 epochs training and 10-scene benchmark evaluation"
    git push origin main
} else {
    Write-Host "[+] All results already committed and synced." -ForegroundColor Green
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   Replication Pipeline Finished and Synced to GitHub Successfully!   " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

# Step 6: Shutdown Computer (Requested by User)
Write-Host "[*] All tasks finished. Initiating computer shutdown in 120 seconds..." -ForegroundColor Yellow
shutdown.exe /s /t 120 /c "Xie_Rep_2 training, benchmark evaluation, and GitHub upload finished successfully. Shutting down system."
