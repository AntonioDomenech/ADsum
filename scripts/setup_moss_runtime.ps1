[CmdletBinding()]
param(
    [switch] $Doctor,
    [switch] $Force,
    [switch] $InstallDiarization,
    [switch] $IAcceptPyannoteCommunity1Terms,
    [switch] $PromptForHuggingFaceToken,
    [switch] $RequireDiarization
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$PythonVersion = "3.12.13"
$UvVersion = "0.12.1"
$UvArchiveSha256 = "8fcb0cb46e1229065e344758980924e569bef5882ef45f46fada8fb24e06b74a"
$TorchIndex = "https://download.pytorch.org/whl/cu128"
$AsrModelId = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
$AsrModelRevision = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
$DiarizationModelId = "pyannote/speaker-diarization-community-1"
$DiarizationModelRevision = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
$PyannoteTermsUrl = "https://huggingface.co/pyannote/speaker-diarization-community-1"
$HuggingFaceTokenUrl = "https://huggingface.co/settings/tokens"

if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw "LOCALAPPDATA is not available. ADsum cannot choose a private runtime folder."
}

# Keep the existing private root so an ADsum v3 preview upgrade can reuse its
# Python and CUDA files. Recordings live elsewhere and are never changed here.
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "ADsum\MossRuntime"
$VenvDirectory = Join-Path $RuntimeRoot ".venv"
$PythonExecutable = Join-Path $VenvDirectory "Scripts\python.exe"
$ManagedPythonDirectory = Join-Path $RuntimeRoot "Python"
$UvDirectory = Join-Path $RuntimeRoot "Bootstrap\uv-$UvVersion"
$UvExecutable = Join-Path $UvDirectory "uv.exe"
$UvArchive = Join-Path $RuntimeRoot "Bootstrap\uv-$UvVersion-windows-x64.zip"
$UvCacheDirectory = Join-Path $RuntimeRoot "Cache\uv"
$AsrModelDirectory = Join-Path $RuntimeRoot "Models\FasterWhisper\large-v3-turbo"
$AsrRevisionMarker = Join-Path $AsrModelDirectory "ADSUM_MODEL_REVISION.txt"
$DiarizationModelDirectory = Join-Path $RuntimeRoot "Models\Pyannote\speaker-diarization-community-1"
$DiarizationRevisionMarker = Join-Path $DiarizationModelDirectory "ADSUM_MODEL_REVISION.txt"
$InstallManifest = Join-Path $RuntimeRoot "install.json"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$LocalSpeechBundleCandidates = @(
    (Join-Path $PSScriptRoot "Moss"),
    (Join-Path $RepositoryRoot "src\ADsum.Desktop\Moss")
)
$LocalSpeechBundleDirectory = $LocalSpeechBundleCandidates | Where-Object {
    (Test-Path -LiteralPath (Join-Path $_ "requirements.txt")) -and
    (Test-Path -LiteralPath (Join-Path $_ "local_speech_worker.py"))
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

function Assert-PrivateRuntimePath {
    param([Parameter(Mandatory)] [string] $Path)

    $ResolvedRuntime = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd("\")
    $ResolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd("\")
    if (-not $ResolvedPath.StartsWith("$ResolvedRuntime\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace a path outside the private ADsum runtime: $ResolvedPath"
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

function Test-ModelSnapshot {
    param(
        [Parameter(Mandatory)] [string] $Directory,
        [Parameter(Mandatory)] [string] $RevisionMarker,
        [Parameter(Mandatory)] [string] $ExpectedRevision,
        [Parameter(Mandatory)] [string[]] $RequiredRelativePaths
    )

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $RevisionMarker -PathType Leaf)) {
        return $false
    }
    if ((Get-Content -LiteralPath $RevisionMarker -Raw).Trim() -ne $ExpectedRevision) {
        return $false
    }
    foreach ($RelativePath in $RequiredRelativePaths) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $RelativePath) -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

function Invoke-LocalSpeechDoctor {
    param([switch] $RequireDiarizationModel)

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
            if ($Installed.schemaVersion -ne 2) {
                $Failures.Add("The private runtime manifest is from an older ADsum transcription engine. Run setup again.")
            }
            if ($Installed.pythonVersion -ne $PythonVersion) {
                $Failures.Add("Installed Python pin is '$($Installed.pythonVersion)', expected '$PythonVersion'.")
            }
            if ($Installed.fasterWhisperVersion -ne "1.2.1") {
                $Failures.Add("Installed faster-whisper version does not match the ADsum pin.")
            }
            if ($Installed.pyannoteAudioVersion -ne "4.0.7") {
                $Failures.Add("Installed pyannote.audio version does not match the ADsum pin.")
            }
            if ($Installed.asrModelRevision -ne $AsrModelRevision) {
                $Failures.Add("Installed large-v3-turbo model revision does not match the ADsum pin.")
            }
            if ($Installed.diarizationInstalled -and
                $Installed.diarizationModelRevision -ne $DiarizationModelRevision) {
                $Failures.Add("Installed Community-1 model revision does not match the ADsum pin.")
            }
        }
        catch {
            $Failures.Add("The install manifest could not be read: $($_.Exception.Message)")
        }
    }

    $WorkerPath = $null
    if ($null -ne $LocalSpeechBundleDirectory) {
        $WorkerPath = Join-Path $LocalSpeechBundleDirectory "local_speech_worker.py"
    }
    else {
        $Failures.Add("The bundled local speech worker was not found beside ADsum or in the source tree.")
    }

    if (Test-Path -LiteralPath $PythonExecutable) {
        $DoctorProgram = @'
import ast
import importlib.metadata
import json
import logging
import os
import pathlib
import sys
import warnings

# Keep diagnostic failures as plain captured text. Windows PowerShell otherwise
# decorates native stderr with command-location metadata that hides the useful
# setup message inside a large NativeCommandError block.
sys.stderr = sys.stdout

expected = {
    "torch": "2.11.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "av": "18.0.0",
    "onnxruntime": "1.28.0",
    "tokenizers": "0.22.2",
    "pyannote.audio": "4.0.7",
    "pyannote.core": "6.0.1",
    "pyannote.database": "6.1.1",
    "pyannote.metrics": "4.1",
    "pyannote.pipeline": "4.0.0",
    "torchcodec": "0.14.0",
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

import torch

torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
if not torch_lib.is_dir():
    raise SystemExit(f"PyTorch's CUDA DLL directory is missing: {torch_lib}")
os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
_dll_handles = []
if hasattr(os, "add_dll_directory"):
    _dll_handles.append(os.add_dll_directory(str(torch_lib)))

import ctranslate2
import faster_whisper  # noqa: F401
with warnings.catch_warnings():
    # ADsum passes preloaded PCM to Community-1. The optional TorchCodec path
    # therefore does not need a machine-wide full-shared FFmpeg installation.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        import pyannote.audio  # noqa: F401
    finally:
        logging.disable(previous_logging_disable)

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot use the NVIDIA CUDA GPU.")
if torch.version.cuda != "12.8":
    raise SystemExit(f"PyTorch reports CUDA {torch.version.cuda}; expected CUDA 12.8.")
if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit("CTranslate2 cannot see an NVIDIA CUDA device.")

asr_path = pathlib.Path(sys.argv[2])
asr_marker = asr_path / "ADSUM_MODEL_REVISION.txt"
asr_revision = sys.argv[3]
asr_required = [
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
]
if not asr_path.is_dir():
    raise SystemExit(f"Pinned faster-whisper model folder is missing: {asr_path}")
if not asr_marker.is_file() or asr_marker.read_text(encoding="utf-8-sig").strip() != asr_revision:
    raise SystemExit("The faster-whisper model revision marker is missing or incorrect.")
for relative_path in asr_required:
    if not (asr_path / relative_path).is_file():
        raise SystemExit(f"The faster-whisper model is incomplete: missing {relative_path}.")
if (asr_path / "model.bin").stat().st_size < 100_000_000:
    raise SystemExit("The faster-whisper model weights are incomplete.")

diarization_path = pathlib.Path(sys.argv[4])
diarization_revision = sys.argv[5]
require_diarization = sys.argv[7].lower() == "true"
diarization_marker = diarization_path / "ADSUM_MODEL_REVISION.txt"
diarization_required = [
    "config.yaml",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
    "segmentation/pytorch_model.bin",
]
diarization_installed = (
    diarization_path.is_dir()
    and diarization_marker.is_file()
    and diarization_marker.read_text(encoding="utf-8-sig").strip() == diarization_revision
    and all((diarization_path / relative_path).is_file() for relative_path in diarization_required)
)
if require_diarization and not diarization_installed:
    raise SystemExit(
        "The gated pyannote Community-1 snapshot is not installed. "
        "Accept its terms and run setup with -InstallDiarization."
    )

worker_path = pathlib.Path(sys.argv[6])
ast.parse(worker_path.read_text(encoding="utf-8-sig"), filename=str(worker_path))

gpu = torch.cuda.get_device_properties(0)
result = {
    "python": actual_python,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "ctranslate2": importlib.metadata.version("ctranslate2"),
    "faster_whisper": importlib.metadata.version("faster-whisper"),
    "pyannote_audio": importlib.metadata.version("pyannote.audio"),
    "gpu": gpu.name,
    "gpu_memory_gib": round(gpu.total_memory / (1024 ** 3), 2),
    "asr_model_path": str(asr_path),
    "diarization_installed": diarization_installed,
    "diarization_model_path": str(diarization_path) if diarization_installed else None,
}
print(json.dumps(result))
'@

        try {
            $DoctorScriptDirectory = Join-Path $RuntimeRoot "Bootstrap\Doctor"
            New-Item -ItemType Directory -Force -Path $DoctorScriptDirectory | Out-Null
            $DoctorScriptPath = Join-Path $DoctorScriptDirectory ("doctor-" + [guid]::NewGuid().ToString("N") + ".py")
            [System.IO.File]::WriteAllText($DoctorScriptPath, $DoctorProgram, [System.Text.UTF8Encoding]::new($false))
            try {
                $RequireText = ([bool]$RequireDiarizationModel).ToString().ToLowerInvariant()
                # pyannote and Torch can write harmless optional-component
                # warnings to stderr even when the doctor succeeds. Capture
                # the two native streams as plain text files so Windows
                # PowerShell does not wrap stderr lines as ErrorRecord objects.
                $DoctorStdoutPath = "$DoctorScriptPath.stdout"
                $DoctorStderrPath = "$DoctorScriptPath.stderr"
                $SavedErrorActionPreference = $ErrorActionPreference
                try {
                    $ErrorActionPreference = "Continue"
                    & $PythonExecutable `
                        $DoctorScriptPath `
                        $PythonVersion `
                        $AsrModelDirectory `
                        $AsrModelRevision `
                        $DiarizationModelDirectory `
                        $DiarizationModelRevision `
                        $WorkerPath `
                        $RequireText `
                        1> $DoctorStdoutPath `
                        2> $DoctorStderrPath
                    $DoctorExitCode = $LASTEXITCODE
                    $DoctorOutputParts = @()
                    if (Test-Path -LiteralPath $DoctorStderrPath) {
                        $DoctorOutputParts += (Get-Content -LiteralPath $DoctorStderrPath -Raw)
                    }
                    if (Test-Path -LiteralPath $DoctorStdoutPath) {
                        $DoctorOutputParts += (Get-Content -LiteralPath $DoctorStdoutPath -Raw)
                    }
                    $DoctorOutput = (($DoctorOutputParts -join [Environment]::NewLine).Trim())
                }
                finally {
                    $ErrorActionPreference = $SavedErrorActionPreference
                    Remove-Item -LiteralPath $DoctorStdoutPath -Force -ErrorAction SilentlyContinue
                    Remove-Item -LiteralPath $DoctorStderrPath -Force -ErrorAction SilentlyContinue
                }
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
        throw "Local speech doctor found $($Failures.Count) problem(s):$([Environment]::NewLine)$($FailureText -join [Environment]::NewLine)"
    }

    Write-Host "Local speech doctor passed. The private faster-whisper runtime and pinned ASR model are ready."
    if (-not (Test-Path -LiteralPath $DiarizationRevisionMarker)) {
        Write-Host "Speaker diarization is not installed. This is expected until Community-1 access is accepted explicitly."
    }
}

if ($InstallDiarization -and -not $IAcceptPyannoteCommunity1Terms) {
    throw "Community-1 is gated. First accept its terms at $PyannoteTermsUrl, then rerun with -InstallDiarization -IAcceptPyannoteCommunity1Terms."
}
if ($IAcceptPyannoteCommunity1Terms -and -not $InstallDiarization) {
    throw "-IAcceptPyannoteCommunity1Terms is only used together with -InstallDiarization."
}
if ($PromptForHuggingFaceToken -and -not $InstallDiarization) {
    throw "-PromptForHuggingFaceToken is only used together with -InstallDiarization."
}
if ($InstallDiarization -and
    [string]::IsNullOrWhiteSpace($env:HF_TOKEN) -and
    -not $PromptForHuggingFaceToken) {
    throw "Community-1 needs a Hugging Face token. Create one at $HuggingFaceTokenUrl, then either set HF_TOKEN only in this PowerShell process or add -PromptForHuggingFaceToken for masked local entry."
}

# Do not let a Hugging Face token leak into uv, Python setup, or a later ADsum
# launch. Keep it only in this script's memory until the gated download starts.
$DiarizationAccessToken = $null
if ($InstallDiarization) {
    if ($PromptForHuggingFaceToken) {
        $SecureToken = Read-Host "Paste the read-only Hugging Face token" -AsSecureString
        $TokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureToken)
        try {
            $DiarizationAccessToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($TokenPointer)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($TokenPointer)
        }
        if ([string]::IsNullOrWhiteSpace($DiarizationAccessToken)) {
            throw "No Hugging Face token was entered."
        }
    }
    else {
        $DiarizationAccessToken = $env:HF_TOKEN
    }
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
}

if ($Doctor) {
    Invoke-LocalSpeechDoctor -RequireDiarizationModel:$RequireDiarization
    exit 0
}

if ($RequireDiarization) {
    throw "-RequireDiarization is a Doctor check. Use it together with -Doctor."
}

if ($null -eq $LocalSpeechBundleDirectory) {
    throw "Cannot find Moss\local_speech_worker.py and Moss\requirements.txt. Run this script from the extracted ADsum release or from the repository."
}

$RequirementsPath = Join-Path $LocalSpeechBundleDirectory "requirements.txt"

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$RuntimeDriveName = [System.IO.Path]::GetPathRoot($RuntimeRoot).TrimEnd("\").TrimEnd(":")
$RuntimeDrive = Get-PSDrive -Name $RuntimeDriveName -ErrorAction SilentlyContinue
if ($null -ne $RuntimeDrive -and $RuntimeDrive.Free -lt 8GB) {
    throw "At least 8 GB of free disk space is required to install Python, CUDA packages, and the local speech models."
}

Get-PinnedUv

$env:UV_PYTHON_INSTALL_DIR = $ManagedPythonDirectory
$env:UV_CACHE_DIR = $UvCacheDirectory
$env:UV_MANAGED_PYTHON = "1"

Write-Host "Installing private Python $PythonVersion under $RuntimeRoot..."
Invoke-NativeCommand $UvExecutable @("python", "install", $PythonVersion, "--install-dir", $ManagedPythonDirectory, "--no-bin")

if ($Force -and (Test-Path -LiteralPath $VenvDirectory)) {
    Assert-PrivateRuntimePath $VenvDirectory
    Remove-Item -LiteralPath $VenvDirectory -Recurse -Force
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    Invoke-NativeCommand $UvExecutable @("venv", $VenvDirectory, "--python", $PythonVersion, "--managed-python", "--seed")
}

Write-Host "Installing pinned CUDA 12.8, faster-whisper 1.2.1, and pyannote.audio 4.0.7 dependencies..."
Invoke-NativeCommand $UvExecutable @(
    "pip", "install",
    "--python", $PythonExecutable,
    "--index-strategy", "unsafe-best-match",
    "--extra-index-url", $TorchIndex,
    "--requirements", $RequirementsPath
)

$AsrRequiredFiles = @(
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json"
)
$AsrSnapshotReady = Test-ModelSnapshot `
    -Directory $AsrModelDirectory `
    -RevisionMarker $AsrRevisionMarker `
    -ExpectedRevision $AsrModelRevision `
    -RequiredRelativePaths $AsrRequiredFiles

if ($Force -or -not $AsrSnapshotReady) {
    New-Item -ItemType Directory -Force -Path $AsrModelDirectory | Out-Null
    Write-Host "Downloading the pinned faster-whisper large-v3-turbo snapshot. This happens only when the snapshot is missing or its pin changes..."
    $DownloadAsrProgram = @'
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
    $DownloadAsrScriptPath = Join-Path $DownloadScriptDirectory ("download-asr-" + [guid]::NewGuid().ToString("N") + ".py")
    [System.IO.File]::WriteAllText($DownloadAsrScriptPath, $DownloadAsrProgram, [System.Text.UTF8Encoding]::new($false))
    try {
        Invoke-NativeCommand $PythonExecutable @(
            $DownloadAsrScriptPath,
            $AsrModelId,
            $AsrModelRevision,
            $AsrModelDirectory
        )
    }
    finally {
        Remove-Item -LiteralPath $DownloadAsrScriptPath -Force -ErrorAction SilentlyContinue
    }
    [System.IO.File]::WriteAllText($AsrRevisionMarker, "$AsrModelRevision$([Environment]::NewLine)", [System.Text.UTF8Encoding]::new($false))
}
else {
    Write-Host "The pinned faster-whisper large-v3-turbo snapshot is already present; no model download is needed."
}

if ($InstallDiarization) {
    $DiarizationRequiredFiles = @(
        "config.yaml",
        "embedding\pytorch_model.bin",
        "plda\plda.npz",
        "plda\xvec_transform.npz",
        "segmentation\pytorch_model.bin"
    )
    $DiarizationSnapshotReady = Test-ModelSnapshot `
        -Directory $DiarizationModelDirectory `
        -RevisionMarker $DiarizationRevisionMarker `
        -ExpectedRevision $DiarizationModelRevision `
        -RequiredRelativePaths $DiarizationRequiredFiles

    if ($Force -or -not $DiarizationSnapshotReady) {
        New-Item -ItemType Directory -Force -Path $DiarizationModelDirectory | Out-Null
        Write-Host "Downloading the pinned Community-1 snapshot after explicit terms acceptance..."
        $DownloadDiarizationProgram = @'
from huggingface_hub import snapshot_download
import os
import sys

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is not available to the gated model downloader.")

snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    token=token,
)
'@
        $DownloadScriptDirectory = Join-Path $RuntimeRoot "Bootstrap\Download"
        New-Item -ItemType Directory -Force -Path $DownloadScriptDirectory | Out-Null
        $DownloadDiarizationScriptPath = Join-Path $DownloadScriptDirectory ("download-diarization-" + [guid]::NewGuid().ToString("N") + ".py")
        [System.IO.File]::WriteAllText($DownloadDiarizationScriptPath, $DownloadDiarizationProgram, [System.Text.UTF8Encoding]::new($false))
        try {
            $env:HF_TOKEN = $DiarizationAccessToken
            Invoke-NativeCommand $PythonExecutable @(
                $DownloadDiarizationScriptPath,
                $DiarizationModelId,
                $DiarizationModelRevision,
                $DiarizationModelDirectory
            )
        }
        finally {
            Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
            $DiarizationAccessToken = $null
            Remove-Item -LiteralPath $DownloadDiarizationScriptPath -Force -ErrorAction SilentlyContinue
        }
        [System.IO.File]::WriteAllText($DiarizationRevisionMarker, "$DiarizationModelRevision$([Environment]::NewLine)", [System.Text.UTF8Encoding]::new($false))
    }
    else {
        Write-Host "The pinned Community-1 snapshot is already present; no gated model download is needed."
    }

    # The already-installed path does not enter the downloader's finally block.
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    $DiarizationAccessToken = $null
}

$DiarizationInstalled = Test-ModelSnapshot `
    -Directory $DiarizationModelDirectory `
    -RevisionMarker $DiarizationRevisionMarker `
    -ExpectedRevision $DiarizationModelRevision `
    -RequiredRelativePaths @(
        "config.yaml",
        "embedding\pytorch_model.bin",
        "plda\plda.npz",
        "plda\xvec_transform.npz",
        "segmentation\pytorch_model.bin"
    )

$ManifestData = [ordered]@{
    schemaVersion = 2
    installedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    pythonVersion = $PythonVersion
    uvVersion = $UvVersion
    torchVersion = "2.11.0+cu128"
    torchaudioVersion = "2.11.0+cu128"
    fasterWhisperVersion = "1.2.1"
    ctranslate2Version = "4.8.1"
    pyannoteAudioVersion = "4.0.7"
    asrModelId = $AsrModelId
    asrModelRevision = $AsrModelRevision
    asrModelDirectory = $AsrModelDirectory
    diarizationInstalled = $DiarizationInstalled
    diarizationModelId = $DiarizationModelId
    diarizationModelRevision = $DiarizationModelRevision
    diarizationModelDirectory = $DiarizationModelDirectory
    runtimeRoot = $RuntimeRoot
    pythonExecutable = $PythonExecutable
}
$ManifestJson = $ManifestData | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($InstallManifest, "$ManifestJson$([Environment]::NewLine)", [System.Text.UTF8Encoding]::new($false))

Invoke-LocalSpeechDoctor -RequireDiarizationModel:$InstallDiarization
Write-Host "ADsum can now transcribe locally with faster-whisper after recording stops. No model was added to the ADsum application folder."
if (-not $DiarizationInstalled) {
    Write-Host "Speaker labels remain unavailable until Community-1 access is accepted and setup is rerun with -InstallDiarization."
}
