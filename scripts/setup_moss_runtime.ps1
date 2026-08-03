[CmdletBinding()]
param(
    [switch] $Doctor,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$PythonVersion = "3.12.13"
$UvVersion = "0.12.1"
$UvArchiveSha256 = "8fcb0cb46e1229065e344758980924e569bef5882ef45f46fada8fb24e06b74a"
$TorchIndex = "https://download.pytorch.org/whl/cu128"
$MossSourceRevision = "0e3d1403fd8f1f1c674e883ece96b9f630794ebe"
$MossSourceUrl = "https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/archive/$MossSourceRevision.zip"
$ModelId = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
$ModelRevision = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"

if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw "LOCALAPPDATA is not available. ADsum cannot choose a private runtime folder."
}

$RuntimeRoot = Join-Path $env:LOCALAPPDATA "ADsum\MossRuntime"
$VenvDirectory = Join-Path $RuntimeRoot ".venv"
$PythonExecutable = Join-Path $VenvDirectory "Scripts\python.exe"
$ManagedPythonDirectory = Join-Path $RuntimeRoot "Python"
$UvDirectory = Join-Path $RuntimeRoot "Bootstrap\uv-$UvVersion"
$UvExecutable = Join-Path $UvDirectory "uv.exe"
$UvArchive = Join-Path $RuntimeRoot "Bootstrap\uv-$UvVersion-windows-x64.zip"
$UvCacheDirectory = Join-Path $RuntimeRoot "Cache\uv"
$ModelDirectory = Join-Path $RuntimeRoot "Models\MOSS\$ModelRevision"
$ModelRevisionMarker = Join-Path $ModelDirectory "ADSUM_MODEL_REVISION.txt"
$InstallManifest = Join-Path $RuntimeRoot "install.json"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$MossBundleCandidates = @(
    (Join-Path $PSScriptRoot "Moss"),
    (Join-Path $RepositoryRoot "src\ADsum.Desktop\Moss")
)
$MossBundleDirectory = $MossBundleCandidates | Where-Object {
    Test-Path -LiteralPath (Join-Path $_ "moss_worker.py")
} | Select-Object -First 1

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)] [string] $FilePath,
        [Parameter(Mandatory)] [string[]] $ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

function Get-PinnedUv {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $UvArchive) | Out-Null

    if (-not (Test-Path -LiteralPath $UvExecutable) -or $Force) {
        if (Test-Path -LiteralPath $UvArchive) {
            $ExistingHash = (Get-FileHash -LiteralPath $UvArchive -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($ExistingHash -ne $UvArchiveSha256 -or $Force) {
                Remove-Item -LiteralPath $UvArchive -Force
            }
        }

        if (-not (Test-Path -LiteralPath $UvArchive)) {
            $UvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
            Write-Host "Downloading the pinned uv $UvVersion bootstrapper..."
            Invoke-WebRequest -UseBasicParsing -Uri $UvUrl -OutFile $UvArchive
        }

        $DownloadedHash = (Get-FileHash -LiteralPath $UvArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($DownloadedHash -ne $UvArchiveSha256) {
            throw "The uv archive checksum does not match the pinned release. Expected $UvArchiveSha256 but found $DownloadedHash."
        }

        New-Item -ItemType Directory -Force -Path $UvDirectory | Out-Null
        Expand-Archive -LiteralPath $UvArchive -DestinationPath $UvDirectory -Force
    }

    if (-not (Test-Path -LiteralPath $UvExecutable)) {
        throw "The pinned uv executable was not found after extraction: $UvExecutable"
    }

    $VersionText = (& $UvExecutable --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $VersionText -notmatch "^uv $([regex]::Escape($UvVersion))\b") {
        throw "Expected uv $UvVersion but found '$VersionText'."
    }
}

function Invoke-MossDoctor {
    $Failures = [System.Collections.Generic.List[string]]::new()

    if (-not (Test-Path -LiteralPath $PythonExecutable)) {
        $Failures.Add("Private Python is missing: $PythonExecutable")
    }

    if (-not (Test-Path -LiteralPath $InstallManifest)) {
        $Failures.Add("The install manifest is missing: $InstallManifest")
    }
    else {
        try {
            $Installed = Get-Content -LiteralPath $InstallManifest -Raw | ConvertFrom-Json
            if ($Installed.pythonVersion -ne $PythonVersion) {
                $Failures.Add("Installed Python pin is '$($Installed.pythonVersion)', expected '$PythonVersion'.")
            }
            if ($Installed.mossSourceRevision -ne $MossSourceRevision) {
                $Failures.Add("Installed OpenMOSS source revision does not match the ADsum pin.")
            }
            if ($Installed.modelRevision -ne $ModelRevision) {
                $Failures.Add("Installed MOSS model revision does not match the ADsum pin.")
            }
        }
        catch {
            $Failures.Add("The install manifest could not be read: $($_.Exception.Message)")
        }
    }

    $WorkerPath = $null
    if ($null -ne $MossBundleDirectory) {
        $WorkerPath = Join-Path $MossBundleDirectory "moss_worker.py"
    }
    else {
        $Failures.Add("The bundled MOSS worker was not found beside ADsum or in the source tree.")
    }

    if (Test-Path -LiteralPath $PythonExecutable) {
        $DoctorProgram = @'
import ast
import importlib.metadata
import json
import pathlib
import sys

expected = {
    "torch": "2.11.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "transformers": "5.13.1",
    "huggingface-hub": "1.26.0",
    "safetensors": "0.8.0",
    "numpy": "2.4.6",
}
actual_python = ".".join(str(part) for part in sys.version_info[:3])
if actual_python != sys.argv[1]:
    raise SystemExit(f"Python {actual_python} is installed; expected {sys.argv[1]}.")
for distribution, wanted in expected.items():
    actual = importlib.metadata.version(distribution)
    if actual != wanted:
        raise SystemExit(f"{distribution} {actual} is installed; expected {wanted}.")

import moss_transcribe_diarize  # noqa: F401
import torch
from transformers import AutoConfig

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot use the NVIDIA CUDA GPU.")
if torch.version.cuda != "12.8":
    raise SystemExit(f"PyTorch reports CUDA {torch.version.cuda}; expected CUDA 12.8.")

model_path = pathlib.Path(sys.argv[2])
marker_path = model_path / "ADSUM_MODEL_REVISION.txt"
if not model_path.is_dir():
    raise SystemExit(f"Pinned model folder is missing: {model_path}")
if not marker_path.is_file() or marker_path.read_text(encoding="utf-8-sig").strip() != sys.argv[3]:
    raise SystemExit("The model revision marker is missing or incorrect.")
if not (model_path / "config.json").is_file():
    raise SystemExit("The model config.json file is missing.")
weight_files = list(model_path.rglob("*.safetensors"))
if not weight_files or sum(path.stat().st_size for path in weight_files) < 100_000_000:
    raise SystemExit("The model weight files are missing or incomplete.")

AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

worker_path = pathlib.Path(sys.argv[4])
ast.parse(worker_path.read_text(encoding="utf-8-sig"), filename=str(worker_path))

gpu = torch.cuda.get_device_properties(0)
result = {
    "python": actual_python,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu.name,
    "gpu_memory_gib": round(gpu.total_memory / (1024 ** 3), 2),
    "model_path": str(model_path),
    "model_weight_files": len(weight_files),
}
print(json.dumps(result))
'@

        try {
            $DoctorScriptDirectory = Join-Path $RuntimeRoot "Bootstrap\Doctor"
            New-Item -ItemType Directory -Force -Path $DoctorScriptDirectory | Out-Null
            $DoctorScriptPath = Join-Path $DoctorScriptDirectory ("doctor-" + [guid]::NewGuid().ToString("N") + ".py")
            [System.IO.File]::WriteAllText($DoctorScriptPath, $DoctorProgram, [System.Text.UTF8Encoding]::new($false))
            try {
                $DoctorOutput = (& $PythonExecutable $DoctorScriptPath $PythonVersion $ModelDirectory $ModelRevision $WorkerPath 2>&1 | Out-String).Trim()
                $DoctorExitCode = $LASTEXITCODE
            }
            finally {
                Remove-Item -LiteralPath $DoctorScriptPath -Force -ErrorAction SilentlyContinue
            }
            if ($DoctorExitCode -ne 0) {
                $Failures.Add("Python/CUDA/model check failed: $DoctorOutput")
            }
            elseif (-not [string]::IsNullOrWhiteSpace($DoctorOutput)) {
                Write-Host $DoctorOutput
            }
        }
        catch {
            $Failures.Add("Python/CUDA/model check could not run: $($_.Exception.Message)")
        }
    }

    if ($Failures.Count -gt 0) {
        $FailureText = $Failures | ForEach-Object { " - $_" }
        throw "MOSS doctor found $($Failures.Count) problem(s):$([Environment]::NewLine)$($FailureText -join [Environment]::NewLine)"
    }

    Write-Host "MOSS doctor passed. The private runtime, CUDA packages, worker, and pinned model snapshot are ready."
}

if ($Doctor) {
    Invoke-MossDoctor
    exit 0
}

if ($null -eq $MossBundleDirectory) {
    throw "Cannot find Moss\moss_worker.py. Run this script from the extracted ADsum release or from the repository."
}

$RequirementsPath = Join-Path $MossBundleDirectory "requirements.txt"
if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    throw "Cannot find the pinned MOSS requirements file: $RequirementsPath"
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$RuntimeDrive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($RuntimeRoot).TrimEnd("\").TrimEnd(":")) -ErrorAction SilentlyContinue
if ($null -ne $RuntimeDrive -and $RuntimeDrive.Free -lt 10GB) {
    throw "At least 10 GB of free disk space is required to install Python, CUDA packages, and the MOSS model."
}

Get-PinnedUv

$env:UV_PYTHON_INSTALL_DIR = $ManagedPythonDirectory
$env:UV_CACHE_DIR = $UvCacheDirectory
$env:UV_MANAGED_PYTHON = "1"

Write-Host "Installing private Python $PythonVersion under $RuntimeRoot..."
Invoke-NativeCommand $UvExecutable @("python", "install", $PythonVersion, "--install-dir", $ManagedPythonDirectory, "--no-bin")

if ($Force -and (Test-Path -LiteralPath $VenvDirectory)) {
    $ResolvedRuntime = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd("\")
    $ResolvedVenv = [System.IO.Path]::GetFullPath($VenvDirectory).TrimEnd("\")
    if (-not $ResolvedVenv.StartsWith("$ResolvedRuntime\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace a virtual environment outside the private ADsum runtime."
    }
    Remove-Item -LiteralPath $VenvDirectory -Recurse -Force
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    Invoke-NativeCommand $UvExecutable @("venv", $VenvDirectory, "--python", $PythonVersion, "--managed-python", "--seed")
}

Write-Host "Installing pinned CUDA 12.8, Transformers, and audio dependencies..."
Invoke-NativeCommand $UvExecutable @(
    "pip", "install",
    "--python", $PythonExecutable,
    "--index-strategy", "unsafe-best-match",
    "--extra-index-url", $TorchIndex,
    "--requirements", $RequirementsPath
)

Write-Host "Installing the audited OpenMOSS source revision $MossSourceRevision..."
Invoke-NativeCommand $UvExecutable @(
    "pip", "install",
    "--python", $PythonExecutable,
    "--index-strategy", "unsafe-best-match",
    "--extra-index-url", $TorchIndex,
    "--reinstall-package", "moss-transcribe-diarize",
    "moss-transcribe-diarize @ $MossSourceUrl"
)

New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null
Write-Host "Downloading the pinned MOSS model snapshot. This is the largest download and happens only once..."
$DownloadProgram = @'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
)
'@
$DownloadScriptDirectory = Join-Path $RuntimeRoot "Bootstrap\Download"
New-Item -ItemType Directory -Force -Path $DownloadScriptDirectory | Out-Null
$DownloadScriptPath = Join-Path $DownloadScriptDirectory ("download-" + [guid]::NewGuid().ToString("N") + ".py")
[System.IO.File]::WriteAllText($DownloadScriptPath, $DownloadProgram, [System.Text.UTF8Encoding]::new($false))
try {
    Invoke-NativeCommand $PythonExecutable @($DownloadScriptPath, $ModelId, $ModelRevision, $ModelDirectory)
}
finally {
    Remove-Item -LiteralPath $DownloadScriptPath -Force -ErrorAction SilentlyContinue
}
[System.IO.File]::WriteAllText($ModelRevisionMarker, "$ModelRevision$([Environment]::NewLine)", [System.Text.UTF8Encoding]::new($false))

$ManifestData = [ordered]@{
    schemaVersion = 1
    installedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    pythonVersion = $PythonVersion
    uvVersion = $UvVersion
    torchVersion = "2.11.0+cu128"
    transformersVersion = "5.13.1"
    mossSourceRevision = $MossSourceRevision
    modelId = $ModelId
    modelRevision = $ModelRevision
    runtimeRoot = $RuntimeRoot
    pythonExecutable = $PythonExecutable
    modelDirectory = $ModelDirectory
}
$ManifestJson = $ManifestData | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($InstallManifest, "$ManifestJson$([Environment]::NewLine)", [System.Text.UTF8Encoding]::new($false))

Invoke-MossDoctor
Write-Host "ADsum can now transcribe locally with MOSS. No model was added to the ADsum application folder."
