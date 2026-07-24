$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$deployEnvironment = Join-Path $repoRoot ".venv-deploy"
$deployPython = Join-Path $deployEnvironment "Scripts\python.exe"
$deployTool = Join-Path $deployEnvironment "Scripts\pyside6-deploy.exe"
$deploySpec = Join-Path $repoRoot "pysidedeploy.spec"
$generatedDeploySpec = Join-Path $deployEnvironment "pysidedeploy.generated.spec"
$deployIcon = Join-Path $deployEnvironment "Lib\site-packages\PySide6\scripts\deploy_lib\pyside_icon.ico"
$expectedExecutable = Join-Path $repoRoot "dist\MIVIMLoRAStudio.dist\MIVIMLoRAStudio.exe"

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

$deployRuntime = & $deployPython -c "import struct, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{struct.calcsize(""P"") * 8}')"
if ($LASTEXITCODE -ne 0 -or $deployRuntime -ne "3.13|64") {
    throw "The deployment environment must use 64-bit Python 3.13. Found: $deployRuntime"
}

& $uv.Source pip install --python $deployPython -r (Join-Path $repoRoot "requirements.txt") "Nuitka==4.0" "pip>=26.0"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install deployment dependencies."
}

Copy-Item -LiteralPath $deploySpec -Destination $generatedDeploySpec -Force

$specContent = Get-Content -LiteralPath $generatedDeploySpec -Raw
$specContent = $specContent -replace '(?m)^project_dir\s*=.*$', "project_dir = $repoRoot"
$specContent = $specContent -replace '(?m)^python_path\s*=.*$', "python_path = $deployPython"
$specContent = $specContent -replace '(?m)^icon\s*=.*$', "icon = $deployIcon"
[System.IO.File]::WriteAllText(
    $generatedDeploySpec,
    $specContent,
    [System.Text.UTF8Encoding]::new($false)
)

$repoPrefix = [System.IO.Path]::GetFullPath($repoRoot).TrimEnd("\") + "\"
$staleBuildTargets = @(
    (Join-Path $repoRoot "dist\MIVIMLoRAStudio.build"),
    (Join-Path $repoRoot "dist\MIVIMLoRAStudio.dist")
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

Push-Location $repoRoot
try {
    & $deployTool --config-file $generatedDeploySpec --force
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
}
finally {
    Pop-Location
}
