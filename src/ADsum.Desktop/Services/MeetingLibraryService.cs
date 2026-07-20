using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;

namespace ADsum.Desktop.Services;

public sealed partial class MeetingLibraryService
{
    public string RootDirectory =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ADsum", "Recordings");

    public IReadOnlyList<MeetingLibraryItem> GetMeetings()
    {
        if (!Directory.Exists(RootDirectory))
        {
            return Array.Empty<MeetingLibraryItem>();
        }

        return Directory
            .EnumerateDirectories(RootDirectory)
            .Select(CreateItem)
            .OrderByDescending(item => item.StartedAt ?? item.LastWriteTime)
            .ToList();
    }

    private static MeetingLibraryItem CreateItem(string directory)
    {
        var info = new DirectoryInfo(directory);
        var folderName = info.Name;
        var (startedAt, topic) = ParseFolderName(folderName);
        return new MeetingLibraryItem(
            topic,
            directory,
            startedAt,
            info.LastWriteTime,
            FindRecordingPath(directory, topic),
            FindTranscriptPath(directory, topic),
            FindNotesPath(directory, topic));
    }

    private static (DateTime? StartedAt, string Topic) ParseFolderName(string folderName)
    {
        var match = TimestampPrefix().Match(folderName);
        if (!match.Success)
        {
            return (null, BeautifyTopic(folderName));
        }

        var stamp = match.Groups["stamp"].Value;
        var format = stamp.Length == 15 ? "yyyyMMdd-HHmmss" : "yyyyMMdd-HHmm";
        var startedAt = DateTime.TryParseExact(
            stamp,
            format,
            CultureInfo.InvariantCulture,
            DateTimeStyles.None,
            out var parsed)
            ? parsed
            : (DateTime?)null;

        var topic = folderName[(match.Length)..].Trim('-');
        return (startedAt, BeautifyTopic(topic));
    }

    private static string BeautifyTopic(string value)
    {
        var text = value.Replace('-', ' ').Replace('_', ' ').Trim();
        if (string.IsNullOrWhiteSpace(text))
        {
            return "Untitled meeting";
        }

        return CultureInfo.CurrentCulture.TextInfo.ToTitleCase(text.ToLowerInvariant());
    }

    private static string? FirstExistingPath(string directory, params string[] fileNames)
    {
        foreach (var fileName in fileNames)
        {
            var path = Path.Combine(directory, fileName);
            if (File.Exists(path))
            {
                return path;
            }
        }

        return null;
    }

    private static string? FindTranscriptPath(string directory, string topic) =>
        FirstExistingPath(
            directory,
            MeetingArtifactStore.TranscriptFileNameForTopic(topic),
            MeetingArtifactStore.LegacyTranscriptFileName)
        ?? FirstMatchingPath(directory, "transcription-*.md");

    private static string? FindRecordingPath(string directory, string topic) =>
        FirstExistingPath(
            directory,
            MeetingArtifactStore.RecordingFileNameForTopic(topic),
            MeetingArtifactStore.RecordingFileName,
            "mixed.wav")
        ?? FirstMatchingPath(directory, "recording-*.wav");

    private static string? FindNotesPath(string directory, string topic) =>
        FirstExistingPath(
            directory,
            MeetingArtifactStore.NotesFileNameForTopic(topic),
            MeetingArtifactStore.LegacyMinutesFileName)
        ?? FirstMatchingPath(directory, "notes-*.md");

    private static string? FirstMatchingPath(string directory, string pattern)
    {
        return Directory
            .EnumerateFiles(directory, pattern)
            .OrderByDescending(File.GetLastWriteTime)
            .FirstOrDefault();
    }

    [GeneratedRegex("^(?<stamp>\\d{8}-\\d{6}|\\d{8}-\\d{4})-?")]
    private static partial Regex TimestampPrefix();
}
