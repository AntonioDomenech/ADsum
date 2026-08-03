# ADsum

ADsum is a cross-platform meeting recorder designed to capture system audio and microphone streams simultaneously, transcribe the conversation, and generate actionable notes. The repository is organised following a modular architecture so the audio engine, orchestration pipeline, transcription backends, and note generators can evolve independently.

## ADsum v3 desktop app

ADsum v3 is a Windows-first .NET desktop app with a WPF UI and native WASAPI audio capture. It records the selected microphone and a WASAPI loopback stream from the selected output device at the same time, so it can capture your headset mic and the meeting audio you hear without taking exclusive control of playback.

Run it from source:

```powershell
dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj
```

The desktop app targets .NET 10 on Windows. Transcription is local: [MOSS-Transcribe-Diarize 0.9B](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) writes the words, timestamps, and anonymous speaker labels such as `Speaker A`, `Speaker B`, and `Speaker C`. Meeting audio is not sent to an OpenAI transcription API.

MOSS runs only after recording has stopped and the user asks ADsum to create a transcript. It is not loaded while a meeting is being recorded, so the recorder keeps the computer's memory, GPU, and processor available for the meeting. Local MOSS jobs run one at a time. ADsum v3 also allows only one recording/transcription-capable process per Windows user and session, preventing a second desktop or `--transcribe-file` process from loading another model behind a meeting.

Install the private MOSS runtime once before the first local transcript:

```powershell
# From an extracted v3 release
.\setup_moss_runtime.ps1

# Or from this repository
.\scripts\setup_moss_runtime.ps1
```

The setup downloads a pinned Python 3.12 runtime, CUDA 12.8 PyTorch packages, the audited OpenMOSS source, and the pinned model snapshot. They are stored under `%LOCALAPPDATA%\ADsum\MossRuntime`; the script does not change the normal system Python or `PATH`. Check an existing installation without changing it:

```powershell
.\setup_moss_runtime.ps1 -Doctor
```

The model is intentionally not embedded in the ADsum release ZIP. This keeps the application download small and makes the model revision explicit. See [the v3 local MOSS guide](docs/v3-local-moss.md) for exact revisions, installation details, the long-meeting design, validation, and troubleshooting.

MOSS advertises contexts up to 90 minutes on larger hardware, but ADsum uses the capacity actually measured on the target 8 GB RTX 5050 laptop: five-minute windows with a 30-second overlap. Each new window advances by 4½ minutes. The shared audio helps ADsum join boundary sentences and map speaker labels, while sequential windows let recordings continue well beyond 90 minutes without loading the whole meeting into GPU memory. Longer windows remain an expert override for GPUs with more memory.

OpenAI remains optional for **meeting notes**, not transcription. A key saved in the app, `ADSUM_OPENAI_API_KEY` / `OPENAI_API_KEY`, or a local `.env` file can be used to create a summary, important points, tasks or next steps, and decisions from the local transcript. Meeting minutes default to `gpt-5.5`; set `ADSUM_OPENAI_NOTES_MODEL=gpt-5.4-mini` for a lower-cost notes model. If the short OpenAI meeting-title request is unavailable, ADsum uses its existing local title fallback.

Build a Windows release artifact:

```powershell
.\scripts\build_windows.ps1
```

The build creates:

- `dist\ADsum-v3.0.0-windows-x64.zip`
- `dist\ADsum-v3.0.0-windows-x64.zip.sha256`

The ZIP is a self-contained Windows bundle that can be attached to a GitHub Release. It contains the .NET runtime, MOSS worker, pinned requirements, setup script, and documentation, but no model weights. The checksum file lets a downloader verify that the ZIP arrived unchanged.

Each meeting is stored under `%LOCALAPPDATA%\ADsum\Recordings` in a folder named `yyyyMMdd-HHmm-topic`. The final topic-named recording is mixed from microphone and system audio with bounded level balancing so quieter room speech is less likely to be masked by louder computer audio. The folder contains:

- `recording-<topic>.wav`
- `transcription-<topic>.md`
- `notes-<topic>.md`

The meeting-topic field is optional. After ADsum transcribes an unnamed meeting, it generates a short topic from the transcript, renames the timestamped recording folder, and gives both the audio and transcript matching topic filenames. A topic entered before recording is preserved and is applied to the files when the transcript is created. Existing meetings that still contain `recording.wav` remain supported. If the short OpenAI naming request is unavailable, ADsum derives a local keyword-based topic so the recording does not remain `Untitled meeting`.

The desktop app also includes a **Library** tab for browsing previous meetings, previewing saved minutes/transcripts, opening the recording or folder directly, creating a local transcript for an older recording, and creating notes from an existing transcript.

Recording has priority over local transcription. ADsum does not start a MOSS job while recording. If the user starts another recording while a local transcript is being created, ADsum stops the worker, preserves its completed chunk checkpoints, waits until recording ends, and then resumes the transcript. Local MOSS work is serialized so two model copies cannot exhaust GPU memory. The existing per-meeting write lock still protects transcripts, notes, and folders from conflicting updates. The status badge and Library continue to show the current processing step.

An audio file containing only digital silence or microscopic PCM rounding residue produces a successful empty transcript without loading MOSS. Non-silent model output must still contain canonical timestamps and speaker labels; malformed speech output is rejected instead of being silently accepted.

## Features

- Dual-channel capture architecture with pluggable audio backends.
- Windows desktop UI with microphone plus WASAPI loopback recording.
- Streaming-friendly recording pipeline that writes directly to disk.
- In-app meeting library for reviewing previous recordings, transcripts, and minutes.
- Storage layer backed by SQLite for recording metadata, transcripts, and notes.
- Local MOSS speaker-aware transcription with timestamps; mock inference is reserved for automated offline tests.
- OpenAI meeting-minutes generation for summaries, discussion points, next steps, and decisions.
- Typer-powered CLI for device discovery, recording, transcription, and note generation.

## Repository layout

```
adsum/
  cli.py                 # Command line interface entry point
  config.py              # Global configuration via environment variables
  logging.py             # Structured logging helpers
  core/
    audio/               # Audio capture abstractions and implementations
    pipeline/            # Recording orchestrator
  data/                  # Pydantic models and SQLite storage helpers
  services/
    transcription/       # Transcription provider interfaces & implementations
    notes/               # Notes generation provider interfaces & implementations
  utils/                 # Shared utilities (audio helpers, task helpers)
```

## Getting started

Install the package in editable mode:

```bash
pip install -e .
```

> **All platforms:** Ensure FFmpeg is installed and available on `PATH`, or set
`ADSUM_FFMPEG_BINARY` to the executable path.

Listing audio devices:

```bash
adsum devices
```

Launching the interactive console UI (recordings are controlled from there):

```bash
adsum ui --mic-device 2 --system-device 5 --transcription-backend openai --notes-backend openai
```

The UI launches from the terminal and lets you start, pause, resume, and stop recordings without additional CLI commands. Each channel is written to `recordings/<session-id>/raw`, a combined track is optionally rendered, and transcription/note generation can be triggered from the interface. Results are stored in `adsum.db`.

### Capturing Bluetooth audio with FFmpeg

ADsum now uses FFmpeg as the default capture engine so Bluetooth sources exposed by the operating system can be recorded reliably. When prompted for the microphone or system device provide an FFmpeg-style input specification using the pattern `<format>:<target>?option=value&...`. Examples:

```
# PulseAudio / PipeWire loopback for a Bluetooth headset
pulse:bluez_source.AA_BB_CC_DD_EE_FF.monitor?sample_rate=48000&channels=2

# Windows DirectShow capture from a Bluetooth microphone
dshow:audio=Bluetooth Headset?sample_rate=48000&channels=1

# Windows WASAPI loopback for the current system output (captures audio even when routed to Bluetooth)
wasapi:default?loopback=1

# macOS AVFoundation input index 1
avfoundation:1?channels=1
```

With WASAPI loopback you can stream the system mix while keeping Bluetooth headsets active—the mic and playback channels remain independent, mirroring tools such as Loom. When FFmpeg does not provide WASAPI support or the driver blocks the capture, ADsum falls back to an internal WASAPI loopback backend powered by the `soundcard` library, so the system output is still captured whenever Windows exposes it.

Additional FFmpeg flags can be added via query parameters. For instance `args=-thread_queue_size 2048` (parsed with shell-style quoting) or `opt_timeout=5` (expanded to `-timeout 5`).

> **Windows note:** when a WASAPI loopback device is selected, ADsum will fall back to the `sounddevice`
> (PortAudio) backend automatically whenever the installed FFmpeg build lacks WASAPI support. No extra
> configuration is required—just pick the speaker you want to capture (for example the Bluetooth output) and
> keep `loopback=1` in the device string.

Use the "Configure environment" menu entry to inspect or update any `ADSUM_` variables directly from the UI. Changes are persisted to your `.env` file for future sessions.

### Managing FFmpeg downloads

Both the console and window interfaces call an internal helper named `ensure_ffmpeg_available`
whenever FFmpeg cannot be found. If `ADSUM_FFMPEG_DOWNLOAD_URL` is set, the helper downloads a
platform-specific archive (the `{platform}` placeholder expands to `windows`, `darwin`, or
`linux`) into `<ADSUM_BASE_DIR>/cache/ffmpeg/<platform>`, extracts the binary, and records its
location in `ADSUM_FFMPEG_BINARY`. You can opt-in to this behaviour from the prompts shown after a
failed recording attempt, or configure it ahead of time via the environment menu.

Prefer to manage FFmpeg manually? Simply leave `ADSUM_FFMPEG_DOWNLOAD_URL` unset. The same prompt
lets you browse for the executable and stores the selection in your `.env` file, ensuring future
sessions keep using your preferred installation.

## Configuration

Environment variables customise behaviour via `pydantic` settings (prefix `ADSUM_`):

- `ADSUM_BASE_DIR`: root directory for recordings (default `recordings/`).
- `ADSUM_DATABASE_PATH`: SQLite database path (default `adsum.db`).
- `ADSUM_SAMPLE_RATE`: Sample rate used for capture (default `16000`).
- `ADSUM_CHANNELS`: Number of channels per capture stream (default `1`).
- `ADSUM_CHUNK_SECONDS`: Preferred chunk duration when streaming (default `1.0`).
- `ADSUM_AUDIO_BACKEND`: Audio engine to use (`ffmpeg`).
- `ADSUM_FFMPEG_BINARY`: Override FFmpeg executable path when the binary is not available on PATH.
  On Windows, ADsum also checks common installation folders such as `C:\\ffmpeg\\bin` and
  `C:\\Program Files\\FFmpeg\\bin`. If FFmpeg still cannot be found, download a build from
  [ffmpeg.org](https://ffmpeg.org/download.html) and either add its `bin` directory to `PATH` or
  point `ADSUM_FFMPEG_BINARY` directly at the `ffmpeg.exe` file. When ADsum cannot locate the
  executable during a recording attempt, both interactive interfaces now offer to download or
  browse for the correct binary and persist it to your `.env` file automatically.
- `ADSUM_FFMPEG_DOWNLOAD_URL`: Optional direct download link used by the automatic bootstrapper.
  The URL may include a `{platform}` placeholder that resolves to `windows`, `darwin`, or `linux`.
  When configured, ADsum caches the retrieved archive or binary under
  `<ADSUM_BASE_DIR>/cache/ffmpeg/<platform>` and records the resulting executable path in
  `ADSUM_FFMPEG_BINARY`. Leave this setting empty if you prefer to manage FFmpeg manually.
- `ADSUM_DEFAULT_MIC_DEVICE`: Preferred microphone device identifier remembered between sessions.
- `ADSUM_DEFAULT_SYSTEM_DEVICE`: Preferred system audio device identifier remembered between sessions.
- `ADSUM_OPENAI_TRANSCRIPTION_MODEL`: Model used by the legacy Python CLI's optional OpenAI transcription backend. The v3 Windows desktop app uses local MOSS.
- `ADSUM_OPENAI_NOTES_MODEL`: Model used for OpenAI meeting minutes. Defaults to `gpt-5.5`; use `gpt-5.4-mini` for lower-cost long-meeting notes.
- `ADSUM_OPENAI_MINUTES_MODEL`: Alias for `ADSUM_OPENAI_NOTES_MODEL`.
- `ADSUM_OPENAI_API_KEY`: Optional API key forwarded to the OpenAI client (falls back to `OPENAI_API_KEY`).
- `ADSUM_OPENAI_MAX_UPLOAD_BYTES`: Maximum payload size used by the legacy Python CLI's optional OpenAI transcription backend.

### Choosing a transcription backend

The legacy Python CLI ships with multiple transcription providers and defaults to a lightweight `dummy` backend that returns placeholder text so automated tests can run offline. When using that CLI, explicitly select a real provider:

- **CLI** – pass `--transcription-backend openai` (or your preferred backend) to `adsum record` or `adsum ui` commands.

If the Python CLI uses the OpenAI provider, make sure an API key is available. The v3 .NET desktop app instead uses the separately installed private MOSS runtime described above.

## Development

Run the unit test suite:

```bash
pytest
```

The dummy services ensure tests do not require external APIs or audio hardware.

## License

Apache 2.0

