Param(
    [Parameter(Mandatory=$true)]
    [string]$NewModelPath,

    [string]$TargetModelPath = "model\emotion_detection_v1.keras"
)

if (-not (Test-Path $NewModelPath)) {
    Write-Error "New model file not found: $NewModelPath"
    exit 1
}

# Ensure target dir exists
$dir = Split-Path $TargetModelPath
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

Copy-Item -Path $NewModelPath -Destination $TargetModelPath -Force
Write-Host "Model replaced: $TargetModelPath"
Write-Host "Call the /reload endpoint or restart the server to load the new model."