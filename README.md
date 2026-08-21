# ADsum

ADsum is a cross-platform meeting recorder designed to capture system audio and microphone streams simultaneously, transcribe the conversation, and generate actionable notes. The repository is organised following a modular architecture so the audio engine, orchestration pipeline, transcription backends, and note generators can evolve independently.

## ADsum v3.2 desktop app

ADsum v3.2 is a Windows-first .NET desktop app with a WPF UI and native WASAPI audio capture. It records the selected microphone and a WASAPI loopback stream from the selected output device at the same time, so it can capture your headset mic and the meeting audio you hear without taking exclusive control of playback.

Run it from source:

```powershell
dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj
```

The desktop app targets .NET 10 on Windows and now lets the user choose a transcription model for each run. The original private local pipeline remains the default: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs multilingual Whisper `large-v3-turbo`, and [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) assigns meeting-global anonymous labels such as `Speaker A`, `Speaker B`, and `Speaker C`. Two opt-in OpenAI choices are also available: `gpt-4o-transcribe-diarize` for cloud transcription with speaker labels, and `gpt-transcribe` for high-accuracy file transcription without built-in speaker diarization. Audio leaves the computer only when the user selects an OpenAI model.

All transcription models run only after recording has stopped. They are not loaded during a meeting, so Teams, Zoom, Meet, and ADsum's recorder retain the computer's memory, GPU, and processor. Local speech jobs run one at a time, and a new recording immediately preempts an older local transcription job.

When a recording stops, ADsum automatically creates `recording-compressed.mp3` beside the original WAV. The WAV is retained unchanged. Every transcription backend, including the local one, reads this verified 32 kbps speech MP3 rather than the much larger WAV. ADsum also checks older meeting folders in the background and creates any missing MP3 sidecars without deleting or replacing their source recordings.

Install the private local speech runtime once before the first transcript. Community-1 is free and local, but its publisher requires the account owner to accept its Hugging Face terms once and provide a read-only token for the download:

```powershell
# From an extracted release; token input is masked
.\setup_moss_runtime.ps1 -InstallDiarization -IAcceptPyannoteCommunity1Terms -PromptForHuggingFaceToken

# Or from this repository
.\scripts\setup_moss_runtime.ps1 -InstallDiarization -IAcceptPyannoteCommunity1Terms -PromptForHuggingFaceToken
```

Accept the terms at the [Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1) and create the token at [Hugging Face settings](https://huggingface.co/settings/tokens). The setup prompt masks the token. It exists only in the setup process and is cleared after the gated download; ADsum does not save it. The pinned runtime and model snapshots are stored under `%LOCALAPPDATA%\ADsum\MossRuntime`; this compatibility folder name is retained from v3.0. The script does not change the system Python or `PATH`.

Check the complete installation without changing it:

```powershell
.\setup_moss_runtime.ps1 -Doctor -RequireDiarization
```

Model weights are intentionally not embedded in the ADsum release ZIP. See [the local speech guide](docs/v3-local-moss.md) for exact revisions, setup, benchmark evidence, validation, and troubleshooting. See [the v3.2 transcription guide](docs/v3.2-transcription-models.md) for model selection, MP3 handling, reusable terms, versioned transcripts, OpenAI-key setup, and command-line examples.

ADsum no longer restarts speaker identity every five minutes. faster-whisper consumes the completed recording using internal VAD and batching, while Community-1 performs one logical whole-meeting diarization and globally clusters voice embeddings before assigning A/B/C. A speaker can leave for a long time and still return under the same label. On the target RTX 5050 laptop, a real 1:30:17 meeting completed ASR plus diarization in about 6 minutes 25 seconds. ADsum places no 20-minute recording or transcription cutoff on longer meetings.

OpenAI remains optional. A key saved in the app, `ADSUM_OPENAI_API_KEY` / `OPENAI_API_KEY`, or a local `.env` file enables the two cloud transcription choices and meeting-note generation. Meeting minutes default to `gpt-5.5`; set `ADSUM_OPENAI_NOTES_MODEL=gpt-5.4-mini` for a lower-cost notes model. If the short OpenAI meeting-title request is unavailable, ADsum uses its existing local title fallback. Set `ADSUM_LOCAL_TOPIC_ONLY=1` to force that local title path and prevent the optional title request from sending a transcript excerpt.

The **General** tab stores a default transcription model and reusable spelling terms such as company names, product names, and abbreviations. Terms are entered one per line and are applied to the local Whisper and `gpt-transcribe` requests. OpenAI's specialized `gpt-4o-transcribe-diarize` endpoint does not accept vocabulary hints, so ADsum states that limitation in the model description and transcript provenance instead of pretending the terms were used.

Build a Windows release artifact:

```powershell
.\scripts\build_windows.ps1
```

The build creates:

- `dist\ADsum-v3.2.0-windows-x64.zip`
- `dist\ADsum-v3.2.0-windows-x64.zip.sha256`

The ZIP is a self-contained Windows bundle that can be attached to a GitHub Release. It contains the .NET runtime, local speech worker, pinned requirements, setup script, and documentation, but no model weights or API keys. The checksum file lets a downloader verify that the ZIP arrived unchanged.

Each meeting is stored under `%LOCALAPPDATA%\ADsum\Recordings` in a folder named `yyyyMMdd-HHmm-topic`. The final topic-named recording is mixed from microphone and system audio with bounded level balancing so quieter room speech is less likely to be masked by louder computer audio. The folder contains:

- `recording-<topic>.wav`
- `recording-compressed.mp3`
- `transcription-<model>-<topic>.md` (one retained file per model)
- `notes-<topic>.md`

The meeting-topic field is optional. After ADsum transcribes an unnamed meeting, it generates a short topic from the transcript and renames the timestamped recording folder and original audio. A topic entered before recording is preserved and is applied to newly created files. Existing meetings that still contain `recording.wav` and older `transcription-<topic>.md` files remain supported. If the short OpenAI naming request is unavailable, ADsum derives a local keyword-based topic so the recording does not remain `Untitled meeting`.

The desktop app also includes a **Library** tab for browsing previous meetings, seeing each recording's duration, previewing saved minutes/transcripts, opening the recording or folder directly, and creating notes from an existing transcript. A meeting can be transcribed again with another model: ADsum writes a separate model-specific Markdown file and keeps the earlier transcript. A transcript selector controls which saved version is previewed, opened, or used to create notes.

Recording has priority over local transcription. If a new recording starts while a transcript is being created, ADsum stops the worker and waits until recording ends. A completed ASR stage is atomically checkpointed, so the resumed job can continue with global diarization instead of rewriting every word. Local speech work is serialized so two model copies cannot exhaust GPU memory. The existing per-meeting write lock still protects transcripts, notes, and folders from conflicting updates. The status badge and Library show the current stage and elapsed time.

Non-silent model output must contain valid timestamps and speaker labels; malformed or incomplete output is rejected rather than silently saved.

## Features

- Dual-channel capture architecture with pluggable audio backends.
- Windows desktop UI with microphone plus WASAPI loopback recording.
- Streaming-friendly recording pipeline that writes directly to disk.
- In-app meeting library for reviewing previous recordings, transcripts, and minutes.
- Storage layer backed by SQLite for recording metadata, transcripts, and notes.
- Selectable local or OpenAI transcription, including two speaker-aware choices and the high-accuracy `gpt-transcribe` file model.
- Verified speech-optimized MP3 sidecars while preserving every original WAV.
- Reusable global spelling terms and retained model-specific transcript versions.
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
- `ADSUM_OPENAI_TRANSCRIPTION_MODEL`: Model used by the legacy Python CLI's optional OpenAI transcription backend. The Windows desktop app stores its selected model in `%LOCALAPPDATA%\ADsum\settings.json` and exposes the choice in the Record, Library, and General tabs.
- `ADSUM_OPENAI_NOTES_MODEL`: Model used for OpenAI meeting minutes. Defaults to `gpt-5.5`; use `gpt-5.4-mini` for lower-cost long-meeting notes.
- `ADSUM_OPENAI_MINUTES_MODEL`: Alias for `ADSUM_OPENAI_NOTES_MODEL`.
- `ADSUM_OPENAI_API_KEY`: Optional API key forwarded to the OpenAI client (falls back to `OPENAI_API_KEY`).
- `ADSUM_OPENAI_MAX_UPLOAD_BYTES`: Maximum payload size used by the legacy Python CLI's optional OpenAI transcription backend.

### Choosing a transcription backend

The Windows desktop offers these model identifiers:

- `local-whisper-pyannote`: private local transcription with whole-meeting speaker labels; this remains the default.
- `gpt-4o-transcribe-diarize`: OpenAI cloud transcription with speaker labels; reusable terms are not supported by this endpoint.
- `gpt-transcribe`: OpenAI cloud file transcription with reusable term hints but no built-in speaker labels.

The same choices are available to release-validation commands. These commands always create or reuse the compressed MP3 first:

```powershell
.\ADsum.exe --transcribe-file "C:\path\meeting.wav" --model gpt-transcribe --result "$env:TEMP\adsum-transcription.json"
.\ADsum.exe --transcribe-meeting "C:\path\meeting-folder" --model local-whisper-pyannote --result "$env:TEMP\adsum-meeting-transcription.json"
.\ADsum.exe --compress-recordings --result "$env:TEMP\adsum-compression.json"
```

The legacy Python CLI separately ships with multiple transcription providers and defaults to a lightweight `dummy` backend that returns placeholder text so automated tests can run offline. When using that CLI, explicitly select a real provider:

- **CLI** – pass `--transcription-backend openai` (or your preferred backend) to `adsum record` or `adsum ui` commands.

If either the Python CLI or Windows desktop uses an OpenAI provider, make sure an API key is available. The Windows app encrypts a key saved through its Settings tab with Windows DPAPI; keys are never included in a release archive. The local desktop choice instead uses the separately installed private speech runtime described above.

## Development

Run the unit test suite:

```bash
pytest
```

The dummy services ensure tests do not require external APIs or audio hardware.

## License

Apache 2.0

