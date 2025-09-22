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

Listing audio devices:

```bash
adsum devices
```

Launching the interactive console UI (recordings are controlled from there):

```bash
adsum ui --mic-device 2 --system-device 5 --transcription-backend openai --notes-backend openai
```

The UI launches from the terminal and lets you start, pause, resume, and stop recordings without additional CLI commands. Each channel is written to `recordings/<session-id>/raw`, a combined track is optionally rendered, and transcription/note generation can be triggered from the interface. Results are stored in `adsum.db`.

Use the "Configure environment" menu entry to inspect or update any `ADSUM_` variables directly from the UI. Changes are persisted to your `.env` file for future sessions.

## Configuration

Environment variables customise behaviour via `pydantic` settings (prefix `ADSUM_`):

- `ADSUM_BASE_DIR`: root directory for recordings (default `recordings/`).
- `ADSUM_DATABASE_PATH`: SQLite database path (default `adsum.db`).
- `ADSUM_SAMPLE_RATE`: Sample rate used for capture (default `16000`).
- `ADSUM_CHANNELS`: Number of channels per capture stream (default `1`).
- `ADSUM_CHUNK_SECONDS`: Preferred chunk duration when streaming (default `1.0`).
- `ADSUM_OPENAI_TRANSCRIPTION_MODEL`: Model used for OpenAI transcription.
- `ADSUM_OPENAI_NOTES_MODEL`: Model used for OpenAI notes/summarisation.

### Choosing a transcription backend

ADsum ships with multiple transcription providers. The CLI and desktop window default to a lightweight `dummy` backend that
returns placeholder text so automated tests can run offline. When you are ready to capture real speech, explicitly pick another
provider:

- **CLI** – pass `--transcription-backend openai` (or your preferred backend) to `adsum record` or `adsum ui` commands.
- **Window UI** – open *Configure environment ▸ Transcription backend* and select a real provider before starting a session.

If the dummy backend is still active when you start recording, both interfaces surface a prominent warning so you can switch to
a real service before relying on the transcripts.

## Development

Run the unit test suite:

```bash
pytest
```

The dummy services ensure tests do not require external APIs or audio hardware.

## License

Apache 2.0

