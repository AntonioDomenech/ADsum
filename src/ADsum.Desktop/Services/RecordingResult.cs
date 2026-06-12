namespace ADsum.Desktop.Services;

public sealed record TrackMetrics(
    string? Path,
    TimeSpan Duration,
    float Peak,
    float Rms);

public sealed record RecordingResult(
    string Name,
    string SessionDirectory,
    TimeSpan Duration,
    string? MicrophonePath,
    string? SystemPath,
    string? MixedPath,
    TrackMetrics Microphone,
    TrackMetrics System,
    TrackMetrics Mixed);
