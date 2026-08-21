using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using NAudio.Wave;

namespace ADsum.Desktop.Services;

public sealed partial class MeetingLibraryService
{
    public MeetingLibraryService(string? rootDirectory = null)
    {
        RootDirectory = rootDirectory ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ADsum",
            "Recordings");
    }

    public string RootDirectory { get; }

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
        var recordingPath = FindRecordingPath(directory, topic);
        var transcriptVersions = FindTranscriptVersions(directory);
        return new MeetingLibraryItem(
            topic,
            directory,
            startedAt,
            info.LastWriteTime,
            recordingPath,
            ReadRecordingDuration(recordingPath),
            FindCompressedRecordingPath(directory),
            transcriptVersions.FirstOrDefault()?.Path,
            FindNotesPath(directory, topic),
            transcriptVersions);
    }

    private static TimeSpan? ReadRecordingDuration(string? recordingPath)
    {
        if (string.IsNullOrWhiteSpace(recordingPath))
        {
            return null;
        }

        try
        {
            using var reader = new WaveFileReader(recordingPath);
            return reader.TotalTime;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
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

    private static string? FindRecordingPath(string directory, string topic) =>
        FirstExistingPath(
            directory,
            MeetingArtifactStore.RecordingFileNameForTopic(topic),
            MeetingArtifactStore.RecordingFileName,
            "mixed.wav")
        ?? FirstMatchingPath(directory, "recording-*.wav");

    private static string? FindCompressedRecordingPath(string directory)
    {
        var path = Path.Combine(directory, AudioCompressionService.CompressedFileName);
        return File.Exists(path) ? path : null;
    }

    private static IReadOnlyList<TranscriptVersion> FindTranscriptVersions(string directory)
    {
        return Directory
            .EnumerateFiles(directory, "transcription*.md")
            .Select(path => new TranscriptVersion(
                ModelIdFromTranscriptFile(path),
                TranscriptionModelCatalog.DisplayNameFor(ModelIdFromTranscriptFile(path)),
                path,
                File.GetLastWriteTime(path)))
            .OrderByDescending(version => version.LastWriteTime)
            .ToList();
    }

    private static string ModelIdFromTranscriptFile(string path)
    {
        var fileName = Path.GetFileName(path);
        foreach (var model in TranscriptionModelCatalog.All)
        {
            if (fileName.StartsWith($"transcription-{model.Id}-", StringComparison.OrdinalIgnoreCase))
            {
                return model.Id;
            }
        }

        return TranscriptionModelCatalog.LegacyId;
    }

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
