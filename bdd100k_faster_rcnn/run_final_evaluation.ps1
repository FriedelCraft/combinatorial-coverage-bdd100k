param(
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [string[]]$Methods = @('generic_coverage', 'adaptive_target_aware'),
    [ValidateSet('night', 'rain')]
    [string]$Target = 'night'
)

$ErrorActionPreference = 'Stop'
$env:TQDM_DISABLE = '1'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$finalManifest = Join-Path $PSScriptRoot 'research_splits\final_evaluation.csv'
if (-not (Test-Path -LiteralPath $finalManifest)) { throw 'Final manifest is missing' }

foreach ($seed in $Seeds) {
    foreach ($method in $Methods) {
        $run = Join-Path $PSScriptRoot "outputs\experiments\benchmark_${Target}_${method}_seed${seed}_steps7500"
        $checkpoint = Join-Path $run 'best_checkpoint.pth'
        $history = Join-Path $run 'history.csv'
        $configPath = Join-Path $run 'run_config.json'
        if (-not (Test-Path -LiteralPath $checkpoint) -or
            -not (Test-Path -LiteralPath $history) -or
            -not (Test-Path -LiteralPath $configPath)) {
            throw "Incomplete training run: $run"
        }
        $last = Import-Csv -LiteralPath $history | Select-Object -Last 1
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if ([int]$last.global_step -ne 7500 -or [int]$config.max_steps -ne 7500 -or
            [int]$config.eval_every_steps -ne 3750 -or
            [int]$config.lr_step_steps -ne 3000 -or
            -not ([string]$config.val_manifest).EndsWith('development.csv')) {
            throw "Run configuration does not match frozen protocol: $run"
        }
        $output = Join-Path $run 'domain_metrics_final_evaluation.csv'
        if (Test-Path -LiteralPath $output) {
            Write-Host "Already scored: $method seed $seed"
            continue
        }
        Write-Host "Final evaluation: $method seed $seed"
        & $python evaluate_domains.py --project-root .. --checkpoint $checkpoint `
            --val-manifest $finalManifest --output $output --batch-size 2 `
            --workers 0 2>&1 | Tee-Object -FilePath (Join-Path $run 'final_evaluation.log')
        if ($LASTEXITCODE -ne 0) { throw "Final evaluation failed: $method seed $seed" }
    }
}
