$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Dotnet = "dotnet"
if (Test-Path "C:\Program Files\dotnet\dotnet.exe") {
    $Dotnet = "C:\Program Files\dotnet\dotnet.exe"
}

$PublishDir = "dist\ADsum-win-x64"
$ZipPath = "dist\ADsum-windows-dotnet.zip"

& $Dotnet publish "src\ADsum.Desktop\ADsum.Desktop.csproj" `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -p:PublishSingleFile=true `
    -p:IncludeNativeLibrariesForSelfExtract=true `
    -p:EnableCompressionInSingleFile=true `
    -o $PublishDir

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -Force -Path "$PublishDir\*" -DestinationPath $ZipPath
Write-Host "Built $ZipPath"
