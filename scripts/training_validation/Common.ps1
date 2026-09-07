Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script:PythonPath = Join-Path $script:RepositoryRoot ".venv\Scripts\python.exe"
$script:ValidationRoot = Join-Path $script:RepositoryRoot "gate_results\training-validation"
$script:ValidationStatePath = Join-Path $script:ValidationRoot "current.json"
$script:RequiredBranch = "experiment/eggroll-es-autopilot"

function Assert-PensivePython {
    if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
        throw "Project Python was not found at $script:PythonPath"
    }
}

function Get-ValidationState {
    if (-not (Test-Path -LiteralPath $script:ValidationStatePath -PathType Leaf)) {
        throw "Validation state is missing. Run 01-preflight.ps1 first."
    }

    $state = Get-Content -Raw -LiteralPath $script:ValidationStatePath | ConvertFrom-Json
    foreach ($requiredProperty in ("git_branch", "git_head", "workspace_fingerprint")) {
        if ($null -eq $state.PSObject.Properties[$requiredProperty]) {
            throw "Validation state predates the workspace guard. Run 01-preflight.ps1 again."
        }
    }
    return $state
}

function Set-ValidationStateValue {
    param(
        [Parameter(Mandatory)] [psobject] $State,
        [Parameter(Mandatory)] [string] $Name,
        [AllowNull()] $Value
    )

    if ($null -eq $State.PSObject.Properties[$Name]) {
        $State | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
        return
    }

    $State.$Name = $Value
}

function Save-ValidationState {
    param([Parameter(Mandatory)] [psobject] $State)

    New-Item -ItemType Directory -Force -Path $script:ValidationRoot | Out-Null
    $temporaryPath = "$script:ValidationStatePath.tmp"
    $State | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryPath -Destination $script:ValidationStatePath -Force
}

function Get-CurrentBranch {
    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $branch) {
        throw "Could not read the current Git branch."
    }
    return $branch
}

function Get-WorkspaceFingerprint {
    $paths = @(git ls-files --cached --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not enumerate the workspace for fingerprinting."
    }

    $entries = foreach ($relativePath in ($paths | Sort-Object -Unique)) {
        $absolutePath = Join-Path $script:RepositoryRoot $relativePath
        if (Test-Path -LiteralPath $absolutePath -PathType Leaf) {
            $fileHash = (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
            "$relativePath`t$fileHash"
        } else {
            "$relativePath`tMISSING"
        }
    }

    $payload = [Text.Encoding]::UTF8.GetBytes(($entries -join "`n"))
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $hasher.ComputeHash($payload)
    } finally {
        $hasher.Dispose()
    }
    return ([BitConverter]::ToString($digest).Replace("-", "").ToLowerInvariant())
}

function Assert-StateMatchesWorkspace {
    param([Parameter(Mandatory)] [psobject] $State)

    $currentBranch = Get-CurrentBranch
    if ($currentBranch -ne $script:RequiredBranch -or $State.git_branch -ne $currentBranch) {
        throw "Expected branch '$script:RequiredBranch', found '$currentBranch'. Run 01-preflight.ps1 on the required branch."
    }

    $currentHead = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the current Git commit."
    }
    if ($State.git_head -ne $currentHead) {
        throw "HEAD changed from $($State.git_head) to $currentHead. Run 01-preflight.ps1 again."
    }

    $currentFingerprint = Get-WorkspaceFingerprint
    if ($State.workspace_fingerprint -ne $currentFingerprint) {
        throw "Tracked or untracked workspace content changed after preflight. Run 01-preflight.ps1 again."
    }
}

function Get-RunDirectory {
    param([Parameter(Mandatory)] [psobject] $State)

    return (Join-Path $script:ValidationRoot $State.run_id)
}
