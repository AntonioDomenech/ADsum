# ADsum

ADsum is a cross-platform meeting recorder designed to capture system audio and microphone streams simultaneously, transcribe the conversation, and generate actionable notes. The repository is organised following a modular architecture so the audio engine, orchestration pipeline, transcription backends, and note generators can evolve independently.

## ADsum v2 desktop app

ADsum v2 adds a Windows-first .NET desktop app with a modern WPF UI and native WASAPI audio capture. It records the selected microphone and a WASAPI loopback stream from the selected output device at the same time, so you can capture your headset mic and the meeting audio you are hearing without taking exclusive control of playback.

Run it from source:

```powershell
dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj
```

The desktop app targets .NET 10 on Windows.

OpenAI transcription can use a key saved in the app, `ADSUM_OPENAI_API_KEY` / `OPENAI_API_KEY`, or a local `.env` file with either of those names. Transcription runs OpenAI speaker diarization on the combined mixed recording and labels voices as `Speaker A`, `Speaker B`, and so on, without assuming the microphone is a single person. ADsum then uses the transcript to create meeting minutes with a summary, important points, tasks or next steps, and decisions. Very long mixed recordings are split into upload-sized WAV chunks before transcription; speaker labels may reset between those local chunks. Meeting minutes default to `gpt-5.5`; set `ADSUM_OPENAI_NOTES_MODEL=gpt-5.4-mini` for a lower-cost notes model.

Build a Windows release artifact:

```powershell
.\scripts\build_windows.ps1
```

The build creates `dist\ADsum-windows-dotnet.zip`, a self-contained Windows bundle that can be attached to a GitHub Release. People downloading that ZIP do not need to install the .NET runtime. See `docs/v2-windows-desktop.md` for the manual audio test checklist.

Each meeting is stored under `%LOCALAPPDATA%\ADsum\Recordings` in a folder named `yyyyMMdd-HHmm-topic`. The folder contains:

- `recording.wav`
- `transcription.md`
- `meeting-minutes.md`

## Features

- Dual-channel capture architecture with pluggable audio backends.
- Windows desktop UI with microphone plus WASAPI loopback recording.
- Streaming-friendly recording pipeline that writes directly to disk.
- Storage layer backed by SQLite for recording metadata, transcripts, and notes.
- Speaker-aware transcription services with OpenAI integration and a lightweight dummy fallback for offline tests.
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
- `ADSUM_OPENAI_TRANSCRIPTION_MODEL`: Model used for OpenAI transcription.
- `ADSUM_OPENAI_NOTES_MODEL`: Model used for OpenAI meeting minutes. Defaults to `gpt-5.5`; use `gpt-5.4-mini` for lower-cost long-meeting notes.
- `ADSUM_OPENAI_MINUTES_MODEL`: Alias for `ADSUM_OPENAI_NOTES_MODEL`.
- `ADSUM_OPENAI_API_KEY`: Optional API key forwarded to the OpenAI client (falls back to `OPENAI_API_KEY`).
- `ADSUM_OPENAI_MAX_UPLOAD_BYTES`: Maximum payload size (default ~24 MiB) before recordings are automatically split for OpenAI uploads.

Recordings that exceed the upload limit are transparently chunked into sequential WAV files before contacting OpenAI, ensuring long meetings are transcribed without manual intervention.

### Choosing a transcription backend

ADsum ships with multiple transcription providers. The CLI and desktop window default to a lightweight `dummy` backend that
returns placeholder text so automated tests can run offline. When you are ready to capture real speech, explicitly pick another
provider:

- **CLI** – pass `--transcription-backend openai` (or your preferred backend) to `adsum record` or `adsum ui` commands.
- **Window UI** – open *Configure environment ▸ Transcription backend* and select a real provider before starting a session.

If you choose one of the OpenAI providers, make sure an API key is available. Set the standard `OPENAI_API_KEY` environment variable
before launching ADsum or configure `ADSUM_OPENAI_API_KEY` via the *Environment* menu so the desktop app can save it to your `.env` file.
If the dummy backend is still active when you start recording, both interfaces surface a prominent warning so you can switch to a
real service before relying on the transcripts.

## Development

Run the unit test suite:

```bash
pytest
```

The dummy services ensure tests do not require external APIs or audio hardware.

## License

Apache 2.0

