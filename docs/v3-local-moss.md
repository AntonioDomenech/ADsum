# ADsum local speech pipeline

> The filename is retained so existing release links keep working. ADsum no
> longer uses MOSS as its primary transcription engine.

ADsum records first and processes afterward:

1. During a meeting, ADsum only records microphone and system audio.
2. After **Stop**, local Whisper writes the words and timestamps.
3. The Whisper model is released from memory.
4. Community-1 examines the complete meeting and decides who spoke when.
5. ADsum joins each timestamped word to a meeting-global `Speaker A`,
   `Speaker B`, `Speaker C`, and so on.

No transcription or speaker model is loaded while recording. Audio is not sent
to OpenAI or to another transcription API.

## Why this replaces the five-minute MOSS design

MOSS took roughly three hours to process the tested 1:30:17 meeting on the
target RTX 5050 laptop. Splitting it into five-minute windows also restarted
local speaker identities at every boundary.

The new jobs are separate:

- `faster-whisper` uses internal VAD and GPU batches to write words quickly.
- Community-1 uses sliding windows internally, but it clusters voice embeddings
  across the whole recording before deciding speaker identity.

Those internal windows are memory containers, not separate meetings. Someone
can be silent for 40 minutes and still return as the same speaker. Community-1
also retains regular overlap-aware output in diagnostics. Exclusive diarization
is used only to attach each Whisper word to one most likely speaker.

## Tested speed

Real input on the target machine:

| Item | Measurement |
|---|---:|
| GPU | NVIDIA GeForce RTX 5050 Laptop, 8 GB |
| Recording | 1:30:17.669 |
| Speech after VAD | 1:26:03.2 |
| Quality ASR inference, beam 5 | 2:19.723 |
| Cached model load | 0:03.844 |
| Quality ASR total | 2:23.567 |
| Timestamped words | 13,302 |
| Last detected speech | 1:30:07.31 |

The faster beam-1 diagnostic took 1:34.985 for inference and 2:02.089 including
its first model setup/load path. ADsum uses the measured quality configuration:
beam 5, `int8_float16`, batch 8, word timestamps, multilingual detection, and
Silero VAD. Batch size falls back from 8 to 4 to 2 only after a CUDA
out-of-memory error.

The complete ASR and Community-1 pipeline was later measured at about 6 minutes
25 seconds for this recording. That is a performance measurement, not a timer:
ADsum continues processing recordings of any duration until they finish or the
user cancels them.

## One-time setup

Community-1 is open, free to run locally, and CC-BY-4.0 licensed, but its model
files are gated. The Hugging Face account owner must:

1. Accept the conditions at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read-only token at <https://huggingface.co/settings/tokens>.
3. Run setup from the extracted release:

```powershell
.\setup_moss_runtime.ps1 -InstallDiarization -IAcceptPyannoteCommunity1Terms -PromptForHuggingFaceToken
```

From a source checkout, use:

```powershell
.\scripts\setup_moss_runtime.ps1 -InstallDiarization -IAcceptPyannoteCommunity1Terms -PromptForHuggingFaceToken
```

The prompt masks the token. The setup script holds it only for the gated
download, removes it before unrelated child processes run, clears it afterward,
and never writes it to the command line, manifest, source tree, or ADsum
settings. Automated setup may instead provide process-only `HF_TOKEN` and omit
`-PromptForHuggingFaceToken`.

If Windows blocks the downloaded script, unblock that file or use a one-process
execution-policy bypass. Do not change the machine-wide policy just for ADsum:

```powershell
Unblock-File .\setup_moss_runtime.ps1
.\setup_moss_runtime.ps1 -InstallDiarization -IAcceptPyannoteCommunity1Terms -PromptForHuggingFaceToken
```

Read-only verification:

```powershell
.\setup_moss_runtime.ps1 -Doctor -RequireDiarization
```

The ASR-only doctor intentionally reports that diarization is absent until the
gated snapshot has been installed.

## Pinned components

| Component | Pin |
|---|---|
| Python | `3.12.13` |
| uv | `0.12.1` |
| PyTorch / torchaudio | `2.11.0+cu128` |
| faster-whisper | `1.2.1` |
| CTranslate2 | `4.8.1` |
| pyannote.audio | `4.0.7` |
| ASR model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| ASR revision | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |
| Diarization model | `pyannote/speaker-diarization-community-1` |
| Diarization revision | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` |

The model aliases and revisions resolve to the exact faster-whisper snapshot
used in the full-length benchmark.

## Private locations

The compatibility runtime root remains:

```text
%LOCALAPPDATA%\ADsum\MossRuntime
```

Important paths:

```text
%LOCALAPPDATA%\ADsum\MossRuntime\.venv\Scripts\python.exe
%LOCALAPPDATA%\ADsum\MossRuntime\Models\FasterWhisper\large-v3-turbo
%LOCALAPPDATA%\ADsum\MossRuntime\Models\Pyannote\speaker-diarization-community-1
%LOCALAPPDATA%\ADsum\MossRuntime\Checkpoints\LocalSpeech
%LOCALAPPDATA%\ADsum\MossRuntime\install.json
```

Recordings stay separate under `%LOCALAPPDATA%\ADsum\Recordings`. Rebuilding
the private runtime must never remove a recording.

## Recording priority and checkpoints

The interface and normal workflow are unchanged: **Record**, **Stop**, then
**Create transcript**.

ADsum refuses to start model work while recording. If a new recording begins
during processing, the Windows worker process is terminated so the meeting gets
the GPU and memory. A completed ASR result is saved atomically with an audio
SHA-256 and settings signature. After recording ends, ADsum can reuse compatible
ASR work and continue with global diarization. A changed WAV or changed ASR
setting cannot reuse a stale checkpoint.

Successful jobs remove their checkpoint folder. Interrupted jobs keep it.

## Timing without a processing cutoff

The worker records inspection, model load, ASR, model release, diarization,
merge, and total wall time. ADsum shows friendly stage names and elapsed time in
the existing status area. There is no 20-minute recording or transcription
cutoff. A job continues until the complete saved recording is processed, the
user cancels it, the app closes, or a new recording temporarily preempts it.

Release acceptance requires a cold complete run on the target laptop with:

- a dense meeting of at least 90 minutes;
- English, Spanish, and code-switching samples;
- simultaneous speakers;
- at least five speakers;
- a speaker who leaves for a long period and returns;
- no CUDA out-of-memory event;
- complete timestamp coverage; and
- complete processing of the entire recording without an app-imposed duration limit.

## Important overlap limitation

Community-1 can report that A and B spoke at the same time. Whisper still emits
one text stream. If the louder voice masks the quieter voice, diarization cannot
invent the missing words. Recovering both conversations would require selective
source separation or a second ASR pass and must remain inside the time budget.

## Developer overrides

- `ADSUM_LOCAL_SPEECH_PYTHON`: private Python executable.
- `ADSUM_LOCAL_SPEECH_WORKER`: `local_speech_worker.py` path.
- `ADSUM_LOCAL_SPEECH_ASR_MODEL`: local faster-whisper snapshot.
- `ADSUM_LOCAL_SPEECH_DIARIZATION_MODEL`: local Community-1 snapshot.
- `ADSUM_LOCAL_SPEECH_LANGUAGE`: `auto`, `en`, `es`, or `mixed`.
- `ADSUM_LOCAL_SPEECH_HOTWORDS`: comma/semicolon list or JSON string array.
- `ADSUM_LOCAL_SPEECH_BATCH_SIZE`: `8`, `4`, or `2`.
- `ADSUM_LOCAL_SPEECH_COMPUTE_TYPE`: defaults to `int8_float16`.
- `ADSUM_LOCAL_TOPIC_ONLY`: forces local meeting-title generation.

The old `ADSUM_MOSS_WORKER` variable is deliberately not used as a worker
fallback because its protocol is incompatible. A few language/hotword/Python
compatibility fallbacks remain for existing v3.0 test setups.

## OpenAI notes remain optional

Local transcription and diarization need no OpenAI key. **Create notes** remains
a separate optional action that can send transcript text to the configured
OpenAI notes model. If notes are not requested, meeting audio and transcription
stay local.
