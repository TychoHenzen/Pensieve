. (Join-Path $PSScriptRoot "Common.ps1")

Assert-PensivePython
$state = Get-ValidationState
if ($state.stability_status -ne "failed") {
    throw "This step requires a failed stability report. If stability passed, run 04-practice.ps1."
}

Push-Location $script:RepositoryRoot
try {
    Assert-StateMatchesWorkspace $state
    if (-not (Test-Path -LiteralPath $state.stability_report -PathType Leaf)) {
        throw "The stability report recorded in validation state is missing."
    }

    $runDirectory = Get-RunDirectory $state
    $reportPath = Join-Path $runDirectory "03-trainability.json"
    $progressPath = Join-Path $runDirectory "03-trainability.jsonl"
    if ((Test-Path -LiteralPath $reportPath) -or (Test-Path -LiteralPath $progressPath)) {
        throw "Trainability outputs already exist for this run. Run 01-preflight.ps1 to start a new run."
    }

    & $script:PythonPath -m train.run_stage0_trainability `
        --stability-report $state.stability_report `
        --final-output $reportPath `
        --progress-output $progressPath
    $commandExitCode = $LASTEXITCODE

    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "The trainability command exited with $commandExitCode and did not write a report."
    }

    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $progressBytes = if (Test-Path -LiteralPath $progressPath) {
        (Get-Item -LiteralPath $progressPath).Length
    } else {
        0
    }
    $evidenceCount = [int] $report.overfit_probe.attempt_count +
        [int] $report.causal_probe_count + [int] $report.arm_count

    Set-ValidationStateValue $state "diagnosis_report" $reportPath
    Set-ValidationStateValue $state "diagnosis_progress" $progressPath
    Set-ValidationStateValue $state "diagnosis_status" $report.overall_status
    Save-ValidationState $state

    if ($evidenceCount -eq 0 -or $progressBytes -eq 0) {
        throw "The trainability runner produced no probe evidence. _run_investigation is still incomplete."
    }

    Write-Host "Diagnosis completed with status '$($report.overall_status)' and exit code $commandExitCode."
    Write-Host "Do not start full training. Use this report as the input to a bounded recalibration proposal."
} finally {
    Pop-Location
}
