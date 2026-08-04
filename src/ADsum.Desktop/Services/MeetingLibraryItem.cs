namespace ADsum.Desktop.Services;

public sealed record MeetingLibraryItem(
    string Topic,
    string DirectoryPath,
    DateTime? StartedAt,
    DateTime LastWriteTime,
    string? RecordingPath,
    TimeSpan? RecordingDuration,
    string? TranscriptPath,
    string? MinutesPath)
{
    public bool HasRecording => !string.IsNullOrWhiteSpace(RecordingPath);

    public bool HasTranscript => !string.IsNullOrWhiteSpace(TranscriptPath);

    public bool HasMinutes => !string.IsNullOrWhiteSpace(MinutesPath);

    public string DisplayName => $"{DateText}  {Topic}";

    public string DateText => StartedAt?.ToString("yyyy-MM-dd HH:mm") ?? LastWriteTime.ToString("yyyy-MM-dd HH:mm");

    public string DurationText => RecordingDuration is { } duration
        ? $"Duration: {FormatDuration(duration)}"
        : HasRecording
            ? "Duration: unavailable"
            : "No recording duration";

    public string FileSummary =>
        $"{(HasRecording ? "audio" : "no audio")} - {(HasTranscript ? "transcript" : "no transcript")} - {(HasMinutes ? "minutes" : "no minutes")}";

    private static string FormatDuration(TimeSpan duration)
    {
        var hours = (long)duration.TotalHours;
        return hours > 0
            ? $"{hours}:{duration.Minutes:00}:{duration.Seconds:00}"
            : $"{(long)duration.TotalMinutes}:{duration.Seconds:00}";
    }
}
