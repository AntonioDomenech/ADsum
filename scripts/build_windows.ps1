$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[build]"

.\.venv\Scripts\python.exe -m PyInstaller `
    --noconfirm `
    --clean `
    --name ADsum `
    --windowed `
    --add-data "adsum\desktop\static;adsum\desktop\static" `
    --collect-all soundcard `
    --collect-all sounddevice `
    --collect-all webview `
    --hidden-import webview `
    --hidden-import webview.platforms.edgechromium `
    --hidden-import clr_loader `
    --hidden-import pythonnet `
    "scripts\adsum_desktop_entry.py"

Compress-Archive -Force -Path "dist\ADsum\*" -DestinationPath "dist\ADsum-windows.zip"
Write-Host "Built dist\ADsum-windows.zip"
