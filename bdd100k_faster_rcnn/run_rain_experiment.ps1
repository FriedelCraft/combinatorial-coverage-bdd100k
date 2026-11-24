param([int[]]$Seeds = @(42, 43, 44, 45, 46))

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach ($seed in $Seeds) {
    $subsetDir = "research_subsets/rain_seed${seed}"
    $summary = Join-Path $subsetDir 'rain_selection_summary_7500.json'
    if (-not (Test-Path -LiteralPath $summary)) { throw "Missing rain subset summary: $summary" }
    Write-Host "Rain study training/development evaluation: seed $seed"
    & (Join-Path $PSScriptRoot 'run_research_benchmark.ps1') `
        -Methods @('adaptive_target_aware', 'rain_count_control') `
        -Seed $seed -SubsetDir $subsetDir -Target rain
    if ($LASTEXITCODE -ne 0) { throw "Rain study failed: seed $seed" }
}
Write-Host 'All rain training runs finished; scoring the existing final partition.'
& (Join-Path $PSScriptRoot 'run_final_evaluation.ps1') `
    -Seeds $Seeds -Methods @('adaptive_target_aware', 'rain_count_control') -Target rain
if ($LASTEXITCODE -ne 0) { throw 'Rain final evaluation failed' }
if ((($Seeds | Sort-Object) -join ',') -eq '42,43,44,45,46') {
    & $python summarize_rain_experiment.py
    if ($LASTEXITCODE -ne 0) { throw 'Rain result summary failed' }
}
