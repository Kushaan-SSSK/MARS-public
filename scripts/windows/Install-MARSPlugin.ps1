param(
    [string]$PackageRoot = "",
    [string]$OpenEphysPluginDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
    $candidate = Join-Path $repoRoot "plugin"

    if (-not (Test-Path -LiteralPath (Join-Path $candidate "MARSSleepScorer.dll"))) {
        $distPackage = Join-Path $repoRoot "dist\MARS_v0.1.0"
        if (Test-Path -LiteralPath (Join-Path $distPackage "MARSSleepScorer.dll")) {
            $candidate = $distPackage
        } elseif (Test-Path -LiteralPath (Join-Path $repoRoot "MARSSleepScorer.dll")) {
            $candidate = $repoRoot
        } elseif (Test-Path -LiteralPath (Join-Path $scriptDir "MARSSleepScorer.dll")) {
            $candidate = $scriptDir
        }
    }

    $PackageRoot = $candidate
}

if ([string]::IsNullOrWhiteSpace($OpenEphysPluginDir)) {
    $OpenEphysPluginDir = Join-Path $env:LOCALAPPDATA "Open Ephys\plugins-api10"
}

$dllPath = Join-Path $PackageRoot "MARSSleepScorer.dll"
$resourcePath = Join-Path $PackageRoot "MARSSleepScorer"

if (-not (Test-Path -LiteralPath $dllPath)) {
    throw "MARSSleepScorer.dll not found in package folder: $PackageRoot"
}
if (-not (Test-Path -LiteralPath $resourcePath)) {
    throw "MARSSleepScorer resource folder not found in package folder: $PackageRoot"
}

New-Item -ItemType Directory -Force -Path $OpenEphysPluginDir | Out-Null

Copy-Item -LiteralPath $dllPath -Destination (Join-Path $OpenEphysPluginDir "MARSSleepScorer.dll") -Force
Copy-Item -LiteralPath $resourcePath -Destination $OpenEphysPluginDir -Recurse -Force

Write-Host ""
Write-Host "MARS installed into:"
Write-Host "  $OpenEphysPluginDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Restart Open Ephys."
Write-Host "  2. Confirm MARS Sleep Scorer and MARS Multi Sleep Scorer appear in the processor list."
Write-Host "  3. Start with Output enabled OFF until channels and scoring look correct."
