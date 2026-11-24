param(
    [int[]]$Seeds = @(43, 44),
    [string[]]$Methods = @('generic_coverage', 'visual_cluster_aware'),
    [int]$Steps = 7500,
    [int]$RefreshSeconds = 15
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
while ($true) {
    Clear-Host
    Write-Host "BDD100K development benchmark — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Each run: $Steps optimizer updates, then development evaluation"
    Write-Host ''
    $rows = foreach ($seed in $Seeds) {
        foreach ($method in $Methods) {
            $run = Join-Path $PSScriptRoot "outputs\experiments\benchmark_night_${method}_seed${seed}_steps${Steps}"
            $history = Join-Path $run 'history.csv'
            $completed = 0
            if (Test-Path -LiteralPath $history) {
                $last = Import-Csv -LiteralPath $history | Select-Object -Last 1
                if ($last) { $completed = [int]$last.global_step }
            }
            $evaluated = Test-Path -LiteralPath (Join-Path $run 'domain_metrics_development.csv')
            [PSCustomObject]@{
                Seed = $seed
                Method = $method
                Checkpoint = "$completed / $Steps"
                DevelopmentEval = if ($evaluated) { 'done' } else { 'pending' }
            }
        }
    }
    $rows | Format-Table -AutoSize
    Write-Host 'GPU:'
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    Write-Host ''
    Write-Host 'Checkpoint counts update at 3,750 and 7,500 steps; GPU use shows activity between them.'
    if (($rows | Where-Object DevelopmentEval -ne 'done').Count -eq 0) {
        Write-Host 'All requested runs are complete.'
        break
    }
    Start-Sleep -Seconds $RefreshSeconds
}
