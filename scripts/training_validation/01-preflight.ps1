. (Join-Path $PSScriptRoot "Common.ps1")

Assert-PensivePython
Push-Location $script:RepositoryRoot
try {
    git fetch --prune origin
    if ($LASTEXITCODE -ne 0) {
        throw "git fetch failed with exit code $LASTEXITCODE"
    }

    git status --short --branch
    if ($LASTEXITCODE -ne 0) {
        throw "git status failed with exit code $LASTEXITCODE"
    }

    $gitBranch = Get-CurrentBranch
    if ($gitBranch -ne $script:RequiredBranch) {
        throw "Expected branch '$script:RequiredBranch', found '$gitBranch'."
    }

    ruff check .
    if ($LASTEXITCODE -ne 0) {
        throw "Ruff failed. Resolve the reported findings before GPU validation."
    }

    & $script:PythonPath -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {
        throw "The full pytest suite failed with exit code $LASTEXITCODE"
    }

    openspec validate --all --strict --no-interactive
    if ($LASTEXITCODE -ne 0) {
        throw "Strict OpenSpec validation failed with exit code $LASTEXITCODE"
    }

    $runId = Get-Date -Format "yyyyMMdd-HHmmss"
    $gitHead = (git rev-parse HEAD).Trim()
    $workspaceFingerprint = Get-WorkspaceFingerprint
    $state = [pscustomobject] [ordered] @{
        schema_version = 1
        run_id = $runId
        git_branch = $gitBranch
        git_head = $gitHead
        workspace_fingerprint = $workspaceFingerprint
        preflight_status = "passed"
        stability_status = $null
        diagnosis_status = $null
        practice_status = $null
        full_training_status = $null
        stage0_gate_status = $null
    }
    New-Item -ItemType Directory -Force -Path (Get-RunDirectory $state) | Out-Null
    Save-ValidationState $state
    Write-Host "Preflight passed. Run 02-stability.ps1 next."
} finally {
    Pop-Location
}
