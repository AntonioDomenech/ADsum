# ADsum v2 Windows Desktop

ADsum v2 is a Windows-first .NET desktop recorder with a WPF UI and native WASAPI audio capture.

OpenAI transcription can use a key saved in the app, `ADSUM_OPENAI_API_KEY` / `OPENAI_API_KEY`, or a local `.env` file with either of those names. Long mixed recordings are split into upload-sized WAV chunks before transcription.

## Run from source

```powershell
dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj
```

## Build a Windows artifact

```powershell
.\scripts\build_windows.ps1
```

The .NET build script creates `dist\ADsum-windows-dotnet.zip`, a self-contained Windows bundle that can be attached to a GitHub Release. People downloading that ZIP do not need to install the .NET runtime.

## Manual recording test

1. Open Windows sound settings and set the headset you want to hear through as the current output.
2. Launch ADsum with `dotnet run --project .\src\ADsum.Desktop\ADsum.Desktop.csproj`, or run `ADsum.exe` from the published ZIP.
3. Select the headset microphone in **Microphone**.
4. Select the same headset or speaker output in **System audio**.
5. Click **Test 6 s**.
6. While the test runs, speak into the microphone and confirm you hear the test tone normally.
7. Check the **Last Recording** panel. `Mic`, `System`, and `Mixed` should all show non-zero peak/RMS values.
8. Click **Transcribe** and confirm the returned text includes at least part of what you said or the test phrase.

For a real online meeting, start playback in Teams/Meet/Zoom first, then click **Record**. ADsum records the selected microphone and WASAPI loopback from the selected output device without taking exclusive control of playback.
