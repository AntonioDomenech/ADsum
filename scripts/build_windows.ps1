[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Version = "3.2.0"
$Root = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
$ProjectPath = Join-Path $Root "src\ADsum.Desktop\ADsum.Desktop.csproj"
$PyprojectPath = Join-Path $Root "pyproject.toml"
$SetupScriptPath = Join-Path $Root "scripts\setup_moss_runtime.ps1"
$MossSourceDirectory = Join-Path $Root "src\ADsum.Desktop\Moss"
$LocalSpeechWorkerPath = Join-Path $MossSourceDirectory "local_speech_worker.py"
$MossRequirementsPath = Join-Path $MossSourceDirectory "requirements.txt"
$V3GuidePath = Join-Path $Root "docs\v3-local-moss.md"
$V32GuidePath = Join-Path $Root "docs\v3.2-transcription-models.md"
$DistDirectory = Join-Path $Root "dist"
$PublishDirectory = Join-Path $DistDirectory (".publish-" + [guid]::NewGuid().ToString("N"))
$ArtifactName = "ADsum-v$Version-windows-x64.zip"
$ZipPath = Join-Path $DistDirectory $ArtifactName
$ShaPath = "$ZipPath.sha256"

$Dotnet = "dotnet"
if (Test-Path -LiteralPath "C:\Program Files\dotnet\dotnet.exe") {
    $Dotnet = "C:\Program Files\dotnet\dotnet.exe"
}

foreach ($requiredPath in @($ProjectPath, $PyprojectPath, $SetupScriptPath, $LocalSpeechWorkerPath, $MossRequirementsPath, $V3GuidePath, $V32GuidePath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required release input is missing: $requiredPath"
    }
}

[xml]$ProjectXml = Get-Content -LiteralPath $ProjectPath -Raw
$ProjectVersion = $ProjectXml.SelectSingleNode("/Project/PropertyGroup/Version")
$ProjectFileVersion = $ProjectXml.SelectSingleNode("/Project/PropertyGroup/FileVersion")
if ($null -eq $ProjectVersion -or $ProjectVersion.InnerText -ne $Version) {
    throw "Release version $Version does not match the project Version."
}
if ($null -eq $ProjectFileVersion -or $ProjectFileVersion.InnerText -ne "$Version.0") {
    throw "Release version $Version does not match the project FileVersion."
}

$PyprojectText = Get-Content -LiteralPath $PyprojectPath -Raw
$PyprojectVersionMatch = [regex]::Match(
    $PyprojectText,
    '(?m)^version\s*=\s*"(?<version>[^"]+)"\s*$')
if (-not $PyprojectVersionMatch.Success -or $PyprojectVersionMatch.Groups["version"].Value -ne $Version) {
    throw "Release version $Version does not match pyproject.toml."
}

New-Item -ItemType Directory -Force -Path $DistDirectory | Out-Null

try {
    & $Dotnet publish $ProjectPath `
        -c Release `
        -r win-x64 `
        --self-contained true `
        -p:PublishSingleFile=true `
        -p:IncludeNativeLibrariesForSelfExtract=true `
        -p:EnableCompressionInSingleFile=true `
        -p:DebugType=None `
        -p:DebugSymbols=false `
        -o $PublishDirectory

    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed with exit code $LASTEXITCODE"
    }

    $PublishedLocalSpeechWorker = Join-Path $PublishDirectory "Moss\local_speech_worker.py"
    $PublishedRequirements = Join-Path $PublishDirectory "Moss\requirements.txt"
    foreach ($publishedPath in @($PublishedLocalSpeechWorker, $PublishedRequirements)) {
        if (-not (Test-Path -LiteralPath $publishedPath)) {
            throw "The local transcription runtime support file was not published: $publishedPath"
        }
    }

    $PublishedExe = Join-Path $PublishDirectory "ADsum.exe"
    $PublishedFileVersion = (Get-Item -LiteralPath $PublishedExe).VersionInfo.FileVersion
    if ($PublishedFileVersion -ne "$Version.0") {
        throw "Published ADsum.exe has FileVersion $PublishedFileVersion; expected $Version.0."
    }

    Copy-Item -LiteralPath $SetupScriptPath -Destination (Join-Path $PublishDirectory "setup_moss_runtime.ps1") -Force
    Copy-Item -LiteralPath $V3GuidePath -Destination (Join-Path $PublishDirectory "v3-local-moss.md") -Force
    Copy-Item -LiteralPath $V32GuidePath -Destination (Join-Path $PublishDirectory "v3.2-transcription-models.md") -Force
    Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination (Join-Path $PublishDirectory "README.md") -Force
    Copy-Item -LiteralPath (Join-Path $Root "LICENSE") -Destination (Join-Path $PublishDirectory "LICENSE") -Force

    $ForbiddenModelFiles = Get-ChildItem -LiteralPath $PublishDirectory -Recurse -File | Where-Object {
        $_.Extension -in @(".safetensors", ".bin", ".pt", ".pth", ".ckpt")
    }
    if ($ForbiddenModelFiles) {
        $ForbiddenList = ($ForbiddenModelFiles.FullName -join [Environment]::NewLine)
        throw "Model weights must not be embedded in the release ZIP. Found:$([Environment]::NewLine)$ForbiddenList"
    }


    $ForbiddenCacheFiles = Get-ChildItem -LiteralPath $PublishDirectory -Recurse -File | Where-Object {
        $_.Extension -eq ".pyc" -or $_.FullName -match "[\\/]__pycache__[\\/]"
    }
    if ($ForbiddenCacheFiles) {
        $ForbiddenList = ($ForbiddenCacheFiles.FullName -join [Environment]::NewLine)
        throw "Python cache files must not be embedded in the release ZIP. Found:$([Environment]::NewLine)$ForbiddenList"
    }

    foreach ($existingArtifact in @($ZipPath, $ShaPath)) {
        if (Test-Path -LiteralPath $existingArtifact) {
            Remove-Item -LiteralPath $existingArtifact -Force
        }
    }

    Compress-Archive -Path (Join-Path $PublishDirectory "*") -DestinationPath $ZipPath -CompressionLevel Optimal

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $EntryNames = @($Archive.Entries | ForEach-Object { $_.FullName.Replace("\", "/") })
        $RequiredEntries = @(
            "ADsum.exe",
            "Moss/local_speech_worker.py",
            "Moss/requirements.txt",
            "setup_moss_runtime.ps1",
            "v3-local-moss.md",
            "v3.2-transcription-models.md",
            "README.md",
            "LICENSE"
        )
        $MissingEntries = @($RequiredEntries | Where-Object { $_ -notin $EntryNames })
        if ($MissingEntries) {
            throw "Release ZIP is missing required entries: $($MissingEntries -join ', ')"
        }

        $ForbiddenEntries = @($EntryNames | Where-Object {
            $_ -match "\.(safetensors|bin|pt|pth|ckpt|gguf)$" -or
            $_ -match "(^|/)__pycache__/" -or
            $_ -match "\.pyc$"
        })
        if ($ForbiddenEntries) {
            throw "Release ZIP contains forbidden entries: $($ForbiddenEntries -join ', ')"
        }
    }
    finally {
        $Archive.Dispose()
    }

    $Hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $HashLine = "$Hash  $ArtifactName$([Environment]::NewLine)"
    [System.IO.File]::WriteAllText($ShaPath, $HashLine, [System.Text.UTF8Encoding]::new($false))

    Write-Host "Built $ZipPath"
    Write-Host "SHA-256 $Hash"
    Write-Host "Checksum file $ShaPath"
}
finally {
    if (Test-Path -LiteralPath $PublishDirectory) {
        Remove-Item -LiteralPath $PublishDirectory -Recurse -Force
    }
}
