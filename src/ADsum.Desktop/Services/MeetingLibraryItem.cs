namespace ADsum.Desktop.Services;

public sealed record MeetingLibraryItem(
    string Topic,
    string DirectoryPath,
    DateTime? StartedAt,
    DateTime LastWriteTime,
    string? RecordingPath,
    TimeSpan? RecordingDuration,
    string? CompressedRecordingPath,
    string? TranscriptPath,
    string? MinutesPath,
    IReadOnlyList<TranscriptVersion> TranscriptVersions)
{
    public bool HasRecording => !string.IsNullOrWhiteSpace(RecordingPath);

    public bool HasTranscript => !string.IsNullOrWhiteSpace(TranscriptPath);

    public bool HasCompressedRecording => !string.IsNullOrWhiteSpace(CompressedRecordingPath);

    public bool HasMinutes => !string.IsNullOrWhiteSpace(MinutesPath);

    public string DisplayName => $"{DateText}  {Topic}";

    public string DateText => StartedAt?.ToString("yyyy-MM-dd HH:mm") ?? LastWriteTime.ToString("yyyy-MM-dd HH:mm");

    public string DurationText => RecordingDuration is { } duration
        ? $"Duration: {FormatDuration(duration)}"
        : HasRecording
            ? "Duration: unavailable"
            : "No recording duration";

    public string FileSummary =>
        $"{(HasRecording ? "audio" : "no audio")} - {(HasCompressedRecording ? "compressed MP3" : "MP3 pending")} - " +
        $"{TranscriptVersions.Count} transcript{(TranscriptVersions.Count == 1 ? "" : "s")} - {(HasMinutes ? "minutes" : "no minutes")}";

    private static string FormatDuration(TimeSpan duration)
    {
        var hours = (long)duration.TotalHours;
        return hours > 0
            ? $"{hours}:{duration.Minutes:00}:{duration.Seconds:00}"
            : $"{(long)duration.TotalMinutes}:{duration.Seconds:00}";
    }
}
