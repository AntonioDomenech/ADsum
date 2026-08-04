# Streaming audio compatibility harness

Run the focused compatibility check from the repository root:

```powershell
dotnet run --project tests/ADsum.Desktop.AudioHarness/ADsum.Desktop.AudioHarness.csproj -c Release
```

The harness generates short deterministic PCM and IEEE-float device recordings,
runs the pre-streaming reference algorithm and `MeetingRecorder.MixWaveFiles`, and
requires the two output WAV files to be byte-for-byte identical. It also requires
the duration, peak, and RMS metrics to match exactly and checks that successful
mixing leaves no temporary files behind.

Run a duration-scaled memory check with:

```powershell
dotnet run --project tests/ADsum.Desktop.AudioHarness/ADsum.Desktop.AudioHarness.csproj -c Release -- --memory-minutes 60
```

This creates a sparse silent WAV of the requested duration, mixes and measures
it, verifies the output duration and silence, and reports the observed private
memory baseline and peak. The fixture is deleted afterward.
