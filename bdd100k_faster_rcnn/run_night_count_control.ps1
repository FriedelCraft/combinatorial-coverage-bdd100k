param([int[]]$Seeds = @(42, 43, 44, 45, 46))

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
foreach ($seed in $Seeds) {
    Write-Host "Night-count control: seed $seed"
    & (Join-Path $PSScriptRoot 'run_research_benchmark.ps1') `
        -Methods @('night_count_control') -Seed $seed `
        -SubsetDir "research_subsets/night_count_control_seed${seed}"
    if ($LASTEXITCODE -ne 0) { throw "Night-count control failed: seed $seed" }
}
