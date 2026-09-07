. (Join-Path $PSScriptRoot "Common.ps1")

Assert-PensivePython
$state = Get-ValidationState
if ($state.stability_status -ne "passed" -or $state.practice_status -ne "passed") {
    throw "Full training requires passing stability and practice steps."
}

Push-Location $script:RepositoryRoot
try {
    Assert-StateMatchesWorkspace $state
    $runDirectory = Get-RunDirectory $state
    $checkpointDirectory = Join-Path $runDirectory "05-full-checkpoints"
    if (Test-Path -LiteralPath $checkpointDirectory) {
        throw "Full checkpoint directory already exists. Run 01-preflight.ps1 to start a new run."
    }

    & $script:PythonPath -m train.run_alternating `
        --epochs 5 `
        --phase-steps 50 `
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
        --eval-problem-count 128 `
        --log-every 50 `
        --stability-report $state.stability_report `
        --save-dir $checkpointDirectory `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Full alternating training failed with exit code $LASTEXITCODE"
    }

    $checkpointPath = Join-Path $checkpointDirectory "epoch-5.ckpt"
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "Full training finished without epoch-5.ckpt."
    }

    Set-ValidationStateValue $state "full_training_status" "passed"
    Set-ValidationStateValue $state "full_checkpoint" $checkpointPath
    Save-ValidationState $state
    Write-Host "Full training finished. Run 06-stage0-gate.ps1 next."
} finally {
    Pop-Location
}
