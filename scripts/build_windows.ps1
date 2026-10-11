param(
    [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildOutput = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
$repoPrefix = [System.IO.Path]::GetFullPath($repoRoot).TrimEnd("\") + "\"
if (-not $buildOutput.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Build output must be inside the repository: $buildOutput"
}
$deployEnvironment = Join-Path $repoRoot ".venv-deploy"
$deployPython = Join-Path $deployEnvironment "Scripts\python.exe"
$deploySpec = Join-Path $repoRoot "pysidedeploy.spec"
$generatedDeploySpec = Join-Path $deployEnvironment "pysidedeploy.generated.spec"
$lockedDeployRequirements = Join-Path $deployEnvironment "deploy-requirements.lock"
$iconGenerator = Join-Path $repoRoot "scripts\generate_windows_icon.py"
$deployIcon = Join-Path $deployEnvironment "mivim-lora-studio.ico"
$expectedExecutable = Join-Path $buildOutput "MIVIMLoRAStudio.dist\MIVIMLoRAStudio.exe"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    throw "uv is required to build the Windows executable."
}

if (-not (Test-Path -LiteralPath $deployPython -PathType Leaf)) {
    & $uv.Source venv --python 3.13 $deployEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.13 deployment environment."
    }
}

$deployRuntime = & $deployPython -c "import struct, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{struct.calcsize(chr(80)) * 8}')"
if ($LASTEXITCODE -ne 0 -or $deployRuntime -ne "3.13|64") {
    throw "The deployment environment must use 64-bit Python 3.13. Found: $deployRuntime"
}

& $uv.Source export `
    --quiet `
    --project $repoRoot `
    --locked `
    --no-dev `
    --group deploy `
    --no-emit-project `
    --format requirements.txt `
    --output-file $lockedDeployRequirements
if ($LASTEXITCODE -ne 0) {
    throw "Could not export the locked deployment dependencies."
}

& $uv.Source pip sync `
    --python $deployPython `
    --require-hashes `
    --strict `
    $lockedDeployRequirements
if ($LASTEXITCODE -ne 0) {
    throw "Could not synchronize the locked deployment dependencies."
}

& $deployPython $iconGenerator $deployIcon
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $deployIcon -PathType Leaf)) {
    throw "Could not generate the MIVIM Windows application icon."
}

Copy-Item -LiteralPath $deploySpec -Destination $generatedDeploySpec -Force

$specContent = Get-Content -LiteralPath $generatedDeploySpec -Raw
$specContent = $specContent -replace '(?m)^project_dir\s*=.*$', "project_dir = $repoRoot"
$specContent = $specContent -replace '(?m)^python_path\s*=.*$', "python_path = $deployPython"
$specContent = $specContent -replace '(?m)^icon\s*=.*$', "icon = $deployIcon"
$specContent = $specContent -replace '(?m)^exec_directory\s*=.*$', "exec_directory = $buildOutput"
[System.IO.File]::WriteAllText(
    $generatedDeploySpec,
    $specContent,
    [System.Text.UTF8Encoding]::new($false)
)

$staleBuildTargets = @(
    (Join-Path $buildOutput "MIVIMLoRAStudio.build"),
    (Join-Path $buildOutput "MIVIMLoRAStudio.dist")
)
foreach ($target in $staleBuildTargets) {
    $resolvedTarget = [System.IO.Path]::GetFullPath($target)
    if (-not $resolvedTarget.StartsWith(
        $repoPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to clean a build target outside the repository: $resolvedTarget"
    }
    if (Test-Path -LiteralPath $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $buildOutput -Force | Out-Null
Push-Location $repoRoot
try {
    # Invoke through the selected interpreter so moved checkouts do not retain
    # a console launcher's stale absolute Python path.
    & $deployPython -c "from PySide6.scripts.pyside_tool import deploy; deploy()" --config-file $generatedDeploySpec --force
    if ($LASTEXITCODE -ne 0) {
        throw "pyside6-deploy exited with code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $expectedExecutable -PathType Leaf)) {
        throw "Deployment did not produce the expected executable: $expectedExecutable"
    }
    $outputDirectory = Split-Path -Parent $expectedExecutable
    $executables = @(Get-ChildItem -LiteralPath $outputDirectory -Filter "*.exe" -File)
    $pythonRuntimes = @(
        Get-ChildItem -LiteralPath $outputDirectory -Filter "python*.dll" -File |
            Where-Object { $_.Name -match '^python\d{3}\.dll$' }
    )
    if ($executables.Count -ne 1 -or $executables[0].Name -ne "MIVIMLoRAStudio.exe") {
        throw "Deployment output contains stale or unexpected executable files."
    }
    if ($pythonRuntimes.Count -ne 1 -or $pythonRuntimes[0].Name -ne "python313.dll") {
        throw "Deployment output does not contain exactly one Python 3.13 runtime."
    }
    foreach ($requiredFile in @(
        (Join-Path $outputDirectory "configs\person_sdxl.yaml"),
        (Join-Path $outputDirectory "workflows\validate_sdxl_api.json")
    )) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "Deployment is missing required application data: $requiredFile"
        }
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $expectedExecutable
    if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        Write-Warning (
            "The local package is not Authenticode-signed. Before distributing " +
            "it to casual users, sign MIVIMLoRAStudio.exe with the MIVIM " +
            "publisher certificate and verify Get-AuthenticodeSignature reports Valid."
        )
    }
}
finally {
    Pop-Location
}
