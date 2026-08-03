# ADsum v3: local MOSS transcription

ADsum v3 records a meeting first and transcribes it afterward. Think of those as two separate jobs:

1. During the meeting, ADsum only captures and writes audio.
2. After the meeting has stopped, the user can ask MOSS to turn that saved audio into text.

The MOSS model is not loaded during recording. This leaves the computer's memory, processor, and GPU available for Teams, Zoom, Meet, and the recording itself.

## What is inside the release

`ADsum-v3.0.0-windows-x64.zip` contains:

- The self-contained .NET 10 Windows application.
- `Moss\moss_worker.py`, the small local bridge between ADsum and MOSS.
- `Moss\requirements.txt`, containing exact Python package versions.
- `setup_moss_runtime.ps1`, which creates the private local runtime.
- This guide, the repository README, and the Apache 2.0 license.

The ZIP deliberately does **not** contain MOSS model weights. The setup script downloads one exact model revision to the user's local application-data folder. This makes the application ZIP smaller and prevents a release from silently containing unknown model files.

## Computer and download requirements

The tested target is 64-bit Windows with:

- An NVIDIA GPU and a driver capable of running the CUDA 12.8 PyTorch build.
- Approximately 8 GB or more of GPU memory for the tested five-minute processing window.
- At least 10 GB of free disk space for Python, PyTorch, package caches, and model files.
- Internet access during the first setup.

ADsum itself remains self-contained. The setup does not add Python to `PATH`, replace a system Python installation, or install packages into a user's normal Python environment.

## Install the private runtime

Extract the complete release ZIP. Do not run the setup script from inside the ZIP preview.

In PowerShell, change into the extracted folder and run:

```powershell
.\setup_moss_runtime.ps1
```

From a source checkout, run:

```powershell
.\scripts\setup_moss_runtime.ps1
```

The first setup is large because it downloads CUDA-enabled PyTorch and the MOSS weights. Later transcripts reuse those local files.

If Windows marks the downloaded script as blocked, unblock this one trusted file and run it again:

```powershell
Unblock-File .\setup_moss_runtime.ps1
.\setup_moss_runtime.ps1
```

The setup ends by running a doctor. The doctor verifies Python, exact package versions, CUDA access, the NVIDIA GPU, the worker's Python syntax, and the pinned model files. It does not run a full transcription or prove how much audio a particular GPU can process; the release validation tests below cover inference capacity separately.

Run the same read-only check later with:

```powershell
.\setup_moss_runtime.ps1 -Doctor
```

Use `-Force` only when the private Python environment needs to be rebuilt:

```powershell
.\setup_moss_runtime.ps1 -Force
```

`-Force` is limited to the private ADsum MOSS environment. It does not remove recordings or a system Python installation.

## Pinned runtime

ADsum v3.0.0 uses these exact components:

| Component | Pin |
|---|---|
| Python | `3.12.13` |
| uv bootstrapper | `0.12.1` |
| PyTorch | `2.11.0+cu128` |
| Torchaudio | `2.11.0+cu128` |
| Transformers | `5.13.1` |
| Hugging Face Hub | `1.26.0` |
| Safetensors | `0.8.0` |
| NumPy | `2.4.6` |
| OpenMOSS source | `0e3d1403fd8f1f1c674e883ece96b9f630794ebe` |
| Model | `OpenMOSS-Team/MOSS-Transcribe-Diarize` |
| Model revision | `e8681d68e7042738ffca8ac8212bc8fcb1131ab8` |

The pinned CUDA packages are resolved using `https://download.pytorch.org/whl/cu128`. The setup downloads the OpenMOSS source from its immutable Git commit and the model from its immutable Hugging Face revision. The worker then loads the local snapshot rather than asking for whatever version happens to be newest.

## Private file locations

The setup stores everything below:

```text
%LOCALAPPDATA%\ADsum\MossRuntime
```

Important paths are:

```text
%LOCALAPPDATA%\ADsum\MossRuntime\.venv\Scripts\python.exe
%LOCALAPPDATA%\ADsum\MossRuntime\Models\MOSS\e8681d68e7042738ffca8ac8212bc8fcb1131ab8
%LOCALAPPDATA%\ADsum\MossRuntime\install.json
```

Recordings remain in their existing location:

```text
%LOCALAPPDATA%\ADsum\Recordings
```

The model directory and recordings are separate. Rebuilding the MOSS runtime must never delete meeting audio.

### Optional environment overrides

The defaults above require no configuration. Developers and testers can override them with:

- `ADSUM_MOSS_PYTHON`: Full path to the private `python.exe`.
- `ADSUM_MOSS_WORKER`: Full path to `moss_worker.py`.
- `ADSUM_MOSS_MODEL_PATH`: Full path to a complete local model snapshot.
- `ADSUM_MOSS_LANGUAGE`: `auto`, `en`, `es`, or `mixed`; the default is `auto`.
- `ADSUM_MOSS_HOTWORDS`: Comma/semicolon-separated terms or a JSON string array.
- `ADSUM_MOSS_CHUNK_SECONDS`: Input-window length from 300 through 1,800 seconds; the tested 8 GB default is 300. Values above 300 are intended for larger GPUs and can cause an out-of-memory failure.
- `ADSUM_MOSS_OVERLAP_SECONDS`: Shared boundary audio from 0 through 600 seconds, and always shorter than the input window; the default is 30.
- `ADSUM_MOSS_ENCODER_BATCH_SIZE`: Number of MOSS's internal 30-second Whisper feature blocks encoded together; the default is 1 to bound temporary encoder memory.

Completed long-audio checkpoints are stored below `%LOCALAPPDATA%\ADsum\MossRuntime\Checkpoints`. Successful jobs remove their checkpoints; interrupted jobs keep them for a retry or post-recording resume.

## What happens when creating a transcript

The visible ADsum workflow remains the same:

1. Select the microphone and system output.
2. Press **Record**.
3. Hold the meeting.
4. Press **Stop**.
5. Press **Create transcript**.

Behind the interface, ADsum does the following:

1. Confirms that recording has stopped.
2. Acquires the user/session-wide ADsum recording-and-MOSS slot. A second v3 desktop, device-test, or `--transcribe-file` process cannot load another model copy.
3. Starts the private Python worker and loads the pinned local model.
4. Processes the completed WAV in sequential windows.
5. Joins timestamps and speaker labels into the existing transcript format.
6. Exits the worker when the job ends, releasing GPU and system memory.

ADsum does not perform live transcription. If a new recording begins while an older meeting is being transcribed, recording wins: ADsum stops the MOSS worker, keeps its completed chunk checkpoints, waits until recording ends, and then resumes the saved job. The model and a live recording therefore do not compete for the GPU or system memory.

The `--transcribe-file` switch is an offline diagnostic path. If it is started before the desktop app, the desktop app asks the user to wait for that offline job rather than opening a second recording-capable process. In the normal desktop workflow, recording and transcription live in the same process, so Record can preempt an active MOSS job.

Before model loading, the worker streams over the WAV and checks for true digital silence or microscopic PCM residue. A peak no larger than 8 out of 32,768 (about -72 dBFS) returns a successful empty transcript and does not construct the model. Any stronger signal continues through normal MOSS inference and strict transcript validation.

## Meetings longer than 90 minutes

MOSS is designed for a single context of up to 90 minutes, but that maximum assumes much more working GPU memory than this 8 GB laptop has. Real capacity tests on the target RTX 5050 found that 15-, 25-, and 30-minute windows exhausted GPU memory. A 7½-minute stress window completed but peaked at 7,704 MiB, leaving too little room for normal desktop use. ADsum therefore uses the five-minute window that completed through its final speech with a measured 4,696 MiB peak:

```text
Window length: 5 minutes
Shared overlap: 30 seconds
Advance:        4 minutes 30 seconds
```

A 95-minute recording is processed like this:

```text
Part 1:  00:00-05:00
Part 2:  04:30-09:30
...
Part 20: 85:30-90:30
Part 21: 90:00-95:00
```

The shared 30 seconds are heard twice by MOSS. ADsum uses that identical audio to avoid losing a sentence at a cut, remove duplicate text, and match local `S01`, `S02`, and `S03` labels to the transcript's global `Speaker A`, `Speaker B`, and `Speaker C` labels.

The parts run one after another, never in parallel. A meeting can therefore be much longer than 90 minutes; ADsum creates more sequential windows and atomically checkpoints each completed part.

Speaker matching across separately processed windows is evidence-based. When the same person speaks in an overlap, their label can usually be connected. If a person is absent from the overlap and returns much later, MOSS does not provide a documented permanent voice identity. ADsum should create a new label instead of guessing and incorrectly merging two different people.

## OpenAI meeting notes remain optional

Local MOSS replaces the OpenAI **transcription** call. It does not replace the existing meeting-notes feature.

After reviewing a local transcript, the user can still press **Create notes**. That separate action may send transcript text to the configured OpenAI notes model to produce:

- A summary.
- Important discussion points.
- Tasks or next steps.
- Decisions.

The OpenAI key field is retained for this optional notes step. Recording and local transcription do not require that key. If no notes are requested, the meeting audio and MOSS transcription stay local.

## Build and verify a release

From the repository root:

```powershell
.\scripts\build_windows.ps1
```

The build produces:

```text
dist\ADsum-v3.0.0-windows-x64.zip
dist\ADsum-v3.0.0-windows-x64.zip.sha256
```

The build fails if the worker, requirements, setup script, or this guide is missing. It also fails if a `.safetensors`, `.bin`, `.pt`, `.pth`, or `.ckpt` model-weight file appears in the publish directory.

Verify the release checksum with:

```powershell
$expected = (Get-Content .\dist\ADsum-v3.0.0-windows-x64.zip.sha256).Split()[0]
$actual = (Get-FileHash .\dist\ADsum-v3.0.0-windows-x64.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Checksum mismatch" }
```

## Manual validation checklist

Before publishing v3.0.0:

1. Extract the release into a new folder.
2. Confirm the ZIP contains `ADsum.exe`, `Moss\moss_worker.py`, `Moss\requirements.txt`, and `setup_moss_runtime.ps1`.
3. Confirm the ZIP contains no model-weight files.
4. Run `setup_moss_runtime.ps1 -Doctor` on a configured test machine; treat this as an installation check, not an inference-capacity test.
5. Run real MOSS inference on a dense five-minute capacity sample. Confirm it reaches speech near the end and retains practical GPU headroom.
6. Start a recording and confirm no MOSS Python process or model memory is active.
7. While that recording is active, try a second v3 `--transcribe-file` process and confirm it exits with the single-instance error without starting Python.
8. Stop recording, create a short transcript, and confirm timestamps and at least two speaker labels.
9. Create a transcript from a digitally silent WAV and confirm it completes empty without loading MOSS.
10. While transcribing a multi-chunk saved meeting, start a new recording. Confirm the complete worker job exits, recording remains smooth, and MOSS resumes from its checkpoint only after Stop.
11. Process a synthetic or consented recording longer than 90 minutes through the same five-minute/30-second plan.
12. Confirm the transcript reaches speech after minute 90, timestamps remain in order, and overlap phrases are not duplicated.
13. Create optional OpenAI notes from that local transcript and confirm the existing notes file is preserved.
14. Close ADsum and confirm no MOSS worker remains running.

## Troubleshooting

### Doctor says private Python is missing

Run the setup without `-Doctor`:

```powershell
.\setup_moss_runtime.ps1
```

### Doctor says CUDA is unavailable

Confirm that Windows sees the NVIDIA GPU and install a driver compatible with CUDA 12.8 PyTorch. Reboot after changing the driver, then run the doctor again. The setup installs CUDA-enabled PyTorch wheels; it does not install or replace the NVIDIA display driver.

### Setup runs out of disk space

Free at least 10 GB on the drive containing `%LOCALAPPDATA%`. Package and model downloads use the private runtime's cache, so both the final files and temporary download data need space.

### Model download was interrupted

Run the same setup command again. Hugging Face's snapshot downloader verifies and resumes the pinned snapshot. If the Python environment itself is damaged, add `-Force`.

### First transcript is slow

The first worker start must initialize CUDA and read the model weights from disk. Later chunks in the same job reuse the loaded model. ADsum intentionally runs only one chunk at a time to stay within laptop memory limits.

### MOSS reports that GPU memory is full

The v3.0.0 default is the five-minute window tested on the 8 GB RTX 5050. Remove any `ADSUM_MOSS_CHUNK_SECONDS` override above 300 and close other GPU-heavy applications before retrying. Completed checkpoints remain available after a failed or interrupted job. The worker reports an error rather than silently saving an incomplete transcript.

### A person receives a new letter later in a long meeting

The voice may not have appeared in the shared overlap between two processing windows. ADsum avoids guessing when there is not enough evidence. A future optional speaker-embedding stage could improve this case without changing the interface.

## Security and licenses

MOSS uses custom model code. ADsum pins both the reviewed OpenMOSS source commit and model revision instead of loading an unspecified latest revision. The model is downloaded only during explicit setup and is loaded from the local snapshot afterward.

ADsum is Apache 2.0 licensed. MOSS-Transcribe-Diarize and its source declare Apache 2.0 licensing at the pinned revisions. Third-party Python and CUDA packages keep their own licenses.
