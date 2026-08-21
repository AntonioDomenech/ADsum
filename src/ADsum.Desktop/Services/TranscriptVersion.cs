namespace ADsum.Desktop.Services;

public sealed record TranscriptVersion(
    string ModelId,
    string ModelName,
    string Path,
    DateTime LastWriteTime)
{
    public string DisplayName => $"{ModelName} - {LastWriteTime:yyyy-MM-dd HH:mm}";

    public override string ToString() => DisplayName;
}
