# ADsum v2 Windows Desktop

ADsum v2 is a Windows-first desktop recorder with a local UI and native audio capture.

## Run from source

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\adsum.exe desktop
```

If the desktop webview is unavailable, use:

```powershell
.\.venv\Scripts\adsum.exe desktop --browser
```

## Build a Windows artifact

```powershell
.\scripts\build_windows.ps1
```

The build script creates `dist\ADsum-windows.zip`, which can be attached to a GitHub Release.

## Manual recording test

1. Open Windows sound settings and set the headset you want to hear through as the current output.
2. Launch ADsum with `.\.venv\Scripts\adsum.exe desktop`.
3. Select the headset microphone in **Microphone**.
4. Select the same headset or speaker output in **System audio**.
5. Click **Test 6 s**.
6. While the test runs, speak into the microphone and confirm you hear the test tone normally.
7. Check the **Last Recording** panel. `Mic`, `System`, and `Mixed` should all show non-zero peak/RMS values.
8. Click **Transcribe** and confirm the returned text includes at least part of what you said or the test phrase.

For a real online meeting, start playback in Teams/Meet/Zoom first, then click **Record**. ADsum records the selected microphone and WASAPI loopback from the selected output device without taking exclusive control of playback.
