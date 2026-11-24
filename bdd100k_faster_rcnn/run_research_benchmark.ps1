param(
    [string[]]$Methods = @('random', 'target_only', 'manual_80pct_night', 'generic_coverage', 'target_aware'),
    [int]$Seed = 42,
    [int]$Steps = 7500,
    [int]$EvalEverySteps = 3750,
    [int]$LrStepSteps = 3000,
    [string]$SubsetDir = 'research_subsets',
    [ValidateSet('night', 'rain')]
    [string]$Target = 'night',
    [ValidateSet('development', 'final_evaluation')]
    [string]$EvaluationSet = 'development'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project .venv is missing.' }
Set-Location -LiteralPath $PSScriptRoot
$env:TQDM_DISABLE = '1'

foreach ($method in $Methods) {
    $manifest = Join-Path $PSScriptRoot (Join-Path $SubsetDir "${Target}_${method}_7500.csv")
    if (-not (Test-Path -LiteralPath $manifest)) { throw "Missing manifest: $manifest" }
    $run = Join-Path $PSScriptRoot "outputs\experiments\benchmark_${Target}_${method}_seed${Seed}_steps${Steps}"
    $lastCheckpoint = Join-Path $run 'last_checkpoint.pth'
    $history = Join-Path $run 'history.csv'
    $completed = 0
    if ((Test-Path -LiteralPath $lastCheckpoint) -and (Test-Path -LiteralPath $history)) {
        $lastRow = Import-Csv -LiteralPath $history | Select-Object -Last 1
        $completed = [int]$lastRow.global_step
    }
    if ($completed -ge $Steps) {
        Write-Host "Using completed checkpoint for $method ($completed updates)"
    } else {
        New-Item -ItemType Directory -Force -Path $run | Out-Null
        $trainArgs = @('train.py', '--project-root', '..', '--train-manifest', $manifest,
            '--val-manifest', 'research_splits\development.csv', '--output-dir', $run,
            '--max-steps', $Steps, '--lr-step-steps', $LrStepSteps,
            '--eval-every-steps', $EvalEverySteps, '--batch-size', 2,
            '--seed', $Seed, '--workers', 0)
        if ($completed -gt 0) { $trainArgs += @('--resume', $lastCheckpoint) }
        & $python @trainArgs 2>&1 | Tee-Object -FilePath (Join-Path $run 'training.log')
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $method" }
    }
    $checkpoint = Join-Path $run 'best_checkpoint.pth'
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
    $evaluationManifest = Join-Path $PSScriptRoot "research_splits\${EvaluationSet}.csv"
    & $python evaluate_domains.py --project-root .. --checkpoint $checkpoint `
        --val-manifest $evaluationManifest `
        --output (Join-Path $run "domain_metrics_${EvaluationSet}.csv") `
        --batch-size 2 --workers 0 2>&1 | Tee-Object -FilePath (Join-Path $run "${EvaluationSet}.log")
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $method" }
}
