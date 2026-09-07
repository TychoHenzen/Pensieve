. (Join-Path $PSScriptRoot "Common.ps1")

Assert-PensivePython
$state = Get-ValidationState
if ($state.stability_status -ne "passed") {
    throw "Practice training requires a passing stability report."
}

Push-Location $script:RepositoryRoot
try {
    Assert-StateMatchesWorkspace $state
    $runDirectory = Get-RunDirectory $state
    $checkpointDirectory = Join-Path $runDirectory "04-practice-checkpoints"
    if (Test-Path -LiteralPath $checkpointDirectory) {
        throw "Practice checkpoint directory already exists. Run 01-preflight.ps1 to start a new run."
    }

    & $script:PythonPath -m train.run_alternating `
        --epochs 1 `
        --phase-steps 8 `
        --problem-count 32 `
        --eval-problem-count 16 `
        --slot-count 16 `
        --num-steps 2 `
        --gradient-lr 0.0001 `
        --eggroll-lr 0.1 `
        --pop-size 128 `
        --sigma 0.001 `
        --rank 4 `
        --variance-weight 1.0 `
        --prompt-alignment-weight 0.1 `
        --eval-batch-size 8 `
        --fitness-batch-size 8 `
        --log-every 8 `
        --stability-report $state.stability_report `
        --save-dir $checkpointDirectory `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Practice training failed with exit code $LASTEXITCODE"
    }

    $checkpointPath = Join-Path $checkpointDirectory "epoch-1.ckpt"
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "Practice training finished without epoch-1.ckpt."
    }

    Set-ValidationStateValue $state "practice_status" "passed"
    Set-ValidationStateValue $state "practice_checkpoint" $checkpointPath
    Save-ValidationState $state
    Write-Host "Practice training passed. Run 05-full-training.ps1 next."
} finally {
    Pop-Location
}
