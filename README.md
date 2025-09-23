# ADsum

ADsum is a cross-platform meeting recorder designed to capture system audio and microphone streams simultaneously, transcribe the conversation, and generate actionable notes. The repository is organised following a modular architecture so the audio engine, orchestration pipeline, transcription backends, and note generators can evolve independently.

## Features

- Dual-channel capture architecture with pluggable audio backends.
- Streaming-friendly recording pipeline that writes directly to disk.
- Storage layer backed by SQLite for recording metadata, transcripts, and notes.
- Transcription services with OpenAI integration and a lightweight dummy fallback for offline tests.
- Note synthesis service that can call OpenAI or fall back to heuristic summarisation.
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

Install the package in editable mode with the audio extras:

```bash
pip install -e .[audio]
```

> **Windows users:** WASAPI loopback capture requires `sounddevice` version 0.4.6 or later, which is included in the `audio` extras.

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

# macOS AVFoundation input index 1
avfoundation:1?channels=1
```

Additional FFmpeg flags can be added via query parameters. For instance `args=-thread_queue_size 2048` (parsed with shell-style quoting) or `opt_timeout=5` (expanded to `-timeout 5`). If you prefer the previous PortAudio backend set `ADSUM_AUDIO_BACKEND=sounddevice`.

Use the "Configure environment" menu entry to inspect or update any `ADSUM_` variables directly from the UI. Changes are persisted to your `.env` file for future sessions.

## Configuration

Environment variables customise behaviour via `pydantic` settings (prefix `ADSUM_`):

- `ADSUM_BASE_DIR`: root directory for recordings (default `recordings/`).
- `ADSUM_DATABASE_PATH`: SQLite database path (default `adsum.db`).
- `ADSUM_SAMPLE_RATE`: Sample rate used for capture (default `16000`).
- `ADSUM_CHANNELS`: Number of channels per capture stream (default `1`).
- `ADSUM_CHUNK_SECONDS`: Preferred chunk duration when streaming (default `1.0`).
- `ADSUM_AUDIO_BACKEND`: Audio engine to use (`ffmpeg` by default, `sounddevice` for the legacy backend).
- `ADSUM_FFMPEG_BINARY`: Override FFmpeg executable path when the binary is not available on PATH.
  On Windows, ADsum also checks common installation folders such as `C:\\ffmpeg\\bin` and
  `C:\\Program Files\\FFmpeg\\bin`. If FFmpeg still cannot be found, download a build from
  [ffmpeg.org](https://ffmpeg.org/download.html) and either add its `bin` directory to `PATH` or
  point `ADSUM_FFMPEG_BINARY` directly at the `ffmpeg.exe` file.
- `ADSUM_DEFAULT_MIC_DEVICE`: Preferred microphone device identifier remembered between sessions.
- `ADSUM_DEFAULT_SYSTEM_DEVICE`: Preferred system audio device identifier remembered between sessions.
- `ADSUM_OPENAI_TRANSCRIPTION_MODEL`: Model used for OpenAI transcription.
- `ADSUM_OPENAI_NOTES_MODEL`: Model used for OpenAI notes/summarisation.
- `ADSUM_OPENAI_API_KEY`: Optional API key forwarded to the OpenAI client (falls back to `OPENAI_API_KEY`).

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

