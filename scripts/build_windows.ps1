$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Dotnet = "dotnet"
if (Test-Path "C:\Program Files\dotnet\dotnet.exe") {
    $Dotnet = "C:\Program Files\dotnet\dotnet.exe"
}

New-Item -ItemType Directory -Force -Path "dist" | Out-Null

$PublishDir = Join-Path "dist" (".publish-" + [guid]::NewGuid().ToString("N"))
$ZipPath = "dist\ADsum-windows-dotnet.zip"

try {
    & $Dotnet publish "src\ADsum.Desktop\ADsum.Desktop.csproj" `
        -c Release `
        -r win-x64 `
        --self-contained true `
        -p:PublishSingleFile=true `
        -p:IncludeNativeLibrariesForSelfExtract=true `
        -p:EnableCompressionInSingleFile=true `
        -o $PublishDir

    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed with exit code $LASTEXITCODE"
    }

    if (Test-Path $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }

    Compress-Archive -Force -Path "$PublishDir\*" -DestinationPath $ZipPath
    Write-Host "Built $ZipPath"
}
finally {
    if (Test-Path $PublishDir) {
        Remove-Item -LiteralPath $PublishDir -Recurse -Force
    }
}
