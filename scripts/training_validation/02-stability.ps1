. (Join-Path $PSScriptRoot "Common.ps1")

Assert-PensivePython
$state = Get-ValidationState
if ($state.preflight_status -ne "passed") {
    throw "Preflight did not pass. Run 01-preflight.ps1 first."
}

Push-Location $script:RepositoryRoot
try {
    Assert-StateMatchesWorkspace $state
    $runDirectory = Get-RunDirectory $state
    $reportPath = Join-Path $runDirectory "02-stability.json"
    $progressPath = Join-Path $runDirectory "02-stability.jsonl"
    if ((Test-Path -LiteralPath $reportPath) -or (Test-Path -LiteralPath $progressPath)) {
        throw "Stability outputs already exist for this run. Run 01-preflight.ps1 to start a new run."
    }

    & $script:PythonPath -m train.run_eggroll_stability `
        --output $reportPath `
        --progress-output $progressPath `
        --slot-count 16 `
        --num-steps 2 `
        --pop-size 128 `
        --sigma 0.001 `
        --lr 0.1 `
        --rank 4 `
        --eval-batch-size 8 `
        --fitness-batch-size 8 `
        --variance-weight 1.0 `
        --prompt-alignment-weight 0.1 `
        --device cuda
    $commandExitCode = $LASTEXITCODE

    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "The stability command exited with $commandExitCode and did not write a report."
    }

    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    Set-ValidationStateValue $state "stability_report" $reportPath
    Set-ValidationStateValue $state "stability_progress" $progressPath
    Set-ValidationStateValue $state "stability_status" $report.status
    Save-ValidationState $state

    if ($commandExitCode -ne 0 -or $report.status -ne "passed") {
        throw "Stability finished with status '$($report.status)'. Run 03-diagnose-failed-stability.ps1 next."
    }

    Write-Host "Stability passed. Skip 03 and run 04-practice.ps1 next."
} finally {
    Pop-Location
}
