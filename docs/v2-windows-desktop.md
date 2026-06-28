# ADsum v2 Windows Desktop

ADsum v2 is a Windows-first .NET desktop recorder with a WPF UI and native WASAPI audio capture.

OpenAI transcription can use a key saved in the app, `ADSUM_OPENAI_API_KEY` / `OPENAI_API_KEY`, or a local `.env` file with either of those names. Transcription runs OpenAI speaker diarization on the combined mixed recording and labels voices as `Speaker A`, `Speaker B`, and so on, without assuming the microphone is a single person. ADsum then uses the transcript to create meeting minutes with a summary, important points, tasks or next steps, and decisions. To preserve speaker labels, ADsum first sends the full recording when it fits the upload limit; if the raw WAV is too large, it creates a temporary compressed upload copy and still tries to send one continuous file. Only recordings that remain too large after compression are split into chunks, where speaker labels may reset between local chunks.

Meeting minutes default to `gpt-5.5`. For lower-cost minutes on long meetings, set `ADSUM_OPENAI_NOTES_MODEL=gpt-5.4-mini` before launching ADsum.

## Run from source

```powershell
dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj
```

## Build a Windows artifact

```powershell
.\scripts\build_windows.ps1
```

The .NET build script creates `dist\ADsum-windows-dotnet.zip`, a self-contained Windows bundle that can be attached to a GitHub Release. People downloading that ZIP do not need to install the .NET runtime.

Meetings are saved under `%LOCALAPPDATA%\ADsum\Recordings\<yyyyMMdd-HHmm-topic>`. The app also shows the exact folder in the **Last Recording** panel after every recording. The final `recording.wav` is mixed from microphone and system audio with bounded level balancing so quieter room speech is less likely to be masked by louder computer audio.

Each meeting folder contains:

- `recording.wav`
- `transcription-<topic>.md`
- `notes-<topic>.md`

Use the **Library** tab to browse previous meetings, preview saved minutes/transcripts, open the recording or folder directly, create a transcript for an older recording, and create notes from an existing transcript.

## Manual recording test

1. Open Windows sound settings and set the headset you want to hear through as the current output.
2. Launch ADsum with `dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj`, or run `ADsum.exe` from the published ZIP.
3. Select the headset microphone in **Microphone**.
4. Select the same headset or speaker output in **System audio**.
5. Click **Test 6 s**.
6. While the test runs, speak into the microphone and confirm you hear the test tone normally.
7. Check the **Last Recording** panel. `Recording` should show non-zero peak/RMS values.
8. Click **Create transcript** and confirm `transcription-<topic>.md` is created in the meeting folder.
9. Click **Create notes** and confirm `notes-<topic>.md` is created in the meeting folder.

For a real online meeting, start playback in Teams/Meet/Zoom first, then click **Record**. ADsum records the selected microphone and WASAPI loopback from the selected output device without taking exclusive control of playback.
