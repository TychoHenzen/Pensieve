. (Join-Path $PSScriptRoot "Common.ps1")

Assert-PensivePython
$state = Get-ValidationState
if ($state.full_training_status -ne "passed") {
    throw "The Stage 0 gate requires a completed full-training checkpoint."
}

Push-Location $script:RepositoryRoot
try {
    Assert-StateMatchesWorkspace $state
    if (-not (Test-Path -LiteralPath $state.full_checkpoint -PathType Leaf)) {
        throw "The full checkpoint recorded in validation state is missing."
    }

    $runDirectory = Get-RunDirectory $state
    $resultsDirectory = Join-Path $runDirectory "06-stage0-gate"
    if (Test-Path -LiteralPath $resultsDirectory) {
        throw "Stage 0 gate results already exist. Run 01-preflight.ps1 to start a new run."
    }

    & $script:PythonPath -m eval.gate.run_gate `
        --checkpoint $state.full_checkpoint `
        --slot-count 16 `
        --num-steps 2 `
        --device cuda `
        --seeds "0,1,2,3,4" `
        --results-dir $resultsDirectory
    $commandExitCode = $LASTEXITCODE

    $reportPath = Join-Path $resultsDirectory "gate_report.json"
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "The Stage 0 gate exited with $commandExitCode and did not write gate_report.json."
    }

    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $gateStatus = if ($report.pass) { "passed" } else { "failed" }
    Set-ValidationStateValue $state "stage0_gate_status" $gateStatus
    Set-ValidationStateValue $state "stage0_gate_report" $reportPath
    Save-ValidationState $state

    if ($commandExitCode -ne 0 -or -not $report.pass) {
        throw "Stage 0 gate failed. Inspect $reportPath"
    }

    Write-Host "Stage 0 gate passed. Report: $reportPath"
} finally {
    Pop-Location
}
