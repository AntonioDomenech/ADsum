namespace ADsum.Desktop.Services;

public sealed record TrackMetrics(
    string? Path,
    TimeSpan Duration,
    float Peak,
    float Rms);

public sealed record RecordingResult(
    string Name,
    string SessionDirectory,
    DateTime StartedAt,
    TimeSpan Duration,
    string? MicrophonePath,
    string? SystemPath,
    string? MixedPath,
    string? TranscriptPath,
    string? MinutesPath,
    TrackMetrics Microphone,
    TrackMetrics System,
    TrackMetrics Mixed);
