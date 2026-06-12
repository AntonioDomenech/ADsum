using System.IO;
using System.Text;

namespace ADsum.Desktop.Services;

public static class MeetingArtifactStore
{
    public const string RecordingFileName = "recording.wav";
    public const string TranscriptFileName = "transcription.md";
    public const string MinutesFileName = "meeting-minutes.md";

    public static RecordingResult SaveTranscript(RecordingResult result, string transcript)
    {
        Directory.CreateDirectory(result.SessionDirectory);
        var path = Path.Combine(result.SessionDirectory, TranscriptFileName);
        File.WriteAllText(path, BuildTranscriptMarkdown(result, transcript), Encoding.UTF8);
        return result with { TranscriptPath = path };
    }

    public static RecordingResult SaveMinutes(RecordingResult result, string minutesMarkdown)
    {
        var topic = ExtractMarkdownTitle(minutesMarkdown) ?? result.Name;
        result = MoveToTopicDirectory(result, topic);
        var path = Path.Combine(result.SessionDirectory, MinutesFileName);
        File.WriteAllText(path, minutesMarkdown.Trim() + Environment.NewLine, Encoding.UTF8);
        return result with { MinutesPath = path };
    }

    public static string Slugify(string value)
    {
        var normalized = value.Trim().ToLowerInvariant();
        var builder = new StringBuilder();
        var previousDash = false;
        foreach (var character in normalized)
        {
            if (char.IsLetterOrDigit(character))
            {
                builder.Append(character);
                previousDash = false;
            }
            else if (!previousDash)
            {
                builder.Append('-');
                previousDash = true;
            }
        }
        return builder.ToString().Trim('-');
    }

    public static string UniqueDirectory(string parent, string baseName, string? currentDirectory = null)
    {
        Directory.CreateDirectory(parent);
        var candidate = Path.Combine(parent, baseName);
        if (SamePath(candidate, currentDirectory) || !Directory.Exists(candidate))
        {
            return candidate;
        }

        for (var index = 2; ; index++)
        {
            candidate = Path.Combine(parent, $"{baseName}-{index}");
            if (SamePath(candidate, currentDirectory) || !Directory.Exists(candidate))
            {
                return candidate;
            }
        }
    }

    private static RecordingResult MoveToTopicDirectory(RecordingResult result, string topic)
    {
        var parent = Directory.GetParent(result.SessionDirectory)?.FullName
            ?? throw new InvalidOperationException("Recording folder has no parent directory.");
        var stamp = result.StartedAt.ToString("yyyyMMdd-HHmm");
        var slug = Slugify(topic);
        if (string.IsNullOrWhiteSpace(slug))
        {
            slug = Slugify(result.Name);
        }
        if (string.IsNullOrWhiteSpace(slug))
        {
            slug = "untitled-meeting";
        }

        var target = UniqueDirectory(parent, $"{stamp}-{slug}", result.SessionDirectory);
        if (SamePath(target, result.SessionDirectory))
        {
            return result;
        }

        Directory.Move(result.SessionDirectory, target);
        return result with
        {
            Name = topic,
            SessionDirectory = target,
            MixedPath = Repath(result.MixedPath, result.SessionDirectory, target),
            TranscriptPath = Repath(result.TranscriptPath, result.SessionDirectory, target),
            MinutesPath = Repath(result.MinutesPath, result.SessionDirectory, target)
        };
    }

    private static string BuildTranscriptMarkdown(RecordingResult result, string transcript)
    {
        var builder = new StringBuilder();
        builder.AppendLine($"# Speaker Transcript - {result.Name}");
        builder.AppendLine();
        builder.AppendLine($"Recorded: {result.StartedAt:yyyy-MM-dd HH:mm}");
        builder.AppendLine($"Duration: {result.Duration.TotalMinutes:F1} minutes");
        builder.AppendLine();
        builder.AppendLine("## Transcript");
        builder.AppendLine();
        builder.AppendLine(transcript.Trim());
        builder.AppendLine();
        return builder.ToString();
    }

    private static string? ExtractMarkdownTitle(string markdown)
    {
        foreach (var line in markdown.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None))
        {
            var trimmed = line.Trim();
            if (trimmed.StartsWith("# ", StringComparison.Ordinal))
            {
                return trimmed[2..].Trim();
            }
        }
        return null;
    }

    private static string? Repath(string? path, string oldRoot, string newRoot)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return path;
        }

        var relative = Path.GetRelativePath(oldRoot, path);
        return Path.Combine(newRoot, relative);
    }

    private static bool SamePath(string? first, string? second)
    {
        if (string.IsNullOrWhiteSpace(first) || string.IsNullOrWhiteSpace(second))
        {
            return false;
        }

        return string.Equals(
            Path.GetFullPath(first).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            Path.GetFullPath(second).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);
    }
}
