param(
    [string]$PackageDirectory = "",
    [string]$TrainerDirectory = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $PackageDirectory) {
    $PackageDirectory = Join-Path $repoRoot "dist\release\MIVIMLoRAStudio.dist"
}
$package = (Resolve-Path -LiteralPath $PackageDirectory).Path
if (-not (Test-Path -LiteralPath (Join-Path $package "MIVIMLoRAStudio.exe"))) {
    throw "Build the Windows application before installing it."
}

$programs = Join-Path $env:LOCALAPPDATA "Programs"
$installDirectory = Join-Path $programs "MIVIM LoRA Studio"
$installedExe = Join-Path $installDirectory "MIVIMLoRAStudio.exe"
$appData = Join-Path $env:LOCALAPPDATA "MIVIM\LoRA Pipeline"
$trainerTarget = Join-Path $appData "trainer\ai-toolkit"

# Refuse to replace files in a running installation. A run in progress must
# finish or be stopped from Studio; installing must not terminate training.
$running = @(Get-Process -Name MIVIMLoRAStudio -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $installedExe })
if ($running.Count) { throw "Close MIVIM LoRA Studio before installing the update." }

New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
& robocopy $package $installDirectory /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Could not install the application files." }

# Keep the selected trainer independent of build and test output folders.
# Copying preserves the existing checkout and avoids breaking an active user.
if ($TrainerDirectory) {
    $trainerSource = (Resolve-Path -LiteralPath $TrainerDirectory).Path
    if (-not (Test-Path -LiteralPath (Join-Path $trainerSource "run.py"))) {
        throw "The selected trainer folder does not contain run.py."
    }
    if ($trainerSource -ne $trainerTarget) {
        New-Item -ItemType Directory -Path $trainerTarget -Force | Out-Null
        $excluded = @(".git", "output") | ForEach-Object { Join-Path $trainerSource $_ }
        & robocopy $trainerSource $trainerTarget /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /XD $excluded | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Could not install the trainer files." }
    }
}

$shell = New-Object -ComObject WScript.Shell
$shortcutFolders = @(
    [Environment]::GetFolderPath("Desktop"),
    [Environment]::GetFolderPath("Programs")
)
foreach ($folder in $shortcutFolders) {
    $shortcut = $shell.CreateShortcut((Join-Path $folder "MIVIM LoRA Studio.lnk"))
    $shortcut.TargetPath = $installedExe
    $shortcut.WorkingDirectory = $installDirectory
    $shortcut.IconLocation = "$installedExe,0"
    $shortcut.Description = "Train and review character LoRAs with ComfyUI"
    $shortcut.Save()
}
Write-Output "Installed: $installedExe"
Write-Output "Open MIVIM LoRA Studio from your desktop or Start menu."
exit 0
