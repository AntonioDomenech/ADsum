namespace ADsum.Desktop.Services;

public sealed record MeetingLibraryItem(
    string Topic,
    string DirectoryPath,
    DateTime? StartedAt,
    DateTime LastWriteTime,
    string? RecordingPath,
    string? TranscriptPath,
    string? MinutesPath)
{
    public bool HasRecording => !string.IsNullOrWhiteSpace(RecordingPath);

    public bool HasTranscript => !string.IsNullOrWhiteSpace(TranscriptPath);

    public bool HasMinutes => !string.IsNullOrWhiteSpace(MinutesPath);

    public string DisplayName => $"{DateText}  {Topic}";

    public string DateText => StartedAt?.ToString("yyyy-MM-dd HH:mm") ?? LastWriteTime.ToString("yyyy-MM-dd HH:mm");

    public string FileSummary =>
        $"{(HasRecording ? "audio" : "no audio")} - {(HasTranscript ? "transcript" : "no transcript")} - {(HasMinutes ? "minutes" : "no minutes")}";
}
