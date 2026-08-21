using System.IO;
using System.Text;

namespace ADsum.Desktop.Services;

public static class MeetingArtifactStore
{
    public const string RecordingFileName = "recording.wav";
    public const string LegacyTranscriptFileName = "transcription.md";
    public const string LegacyMinutesFileName = "meeting-minutes.md";
    private const int MaxSlugLength = 80;

    public static RecordingResult SaveTranscript(RecordingResult result, string transcript, string? generatedTopic = null)
    {
        return SaveTranscript(
            result,
            transcript,
            TranscriptionModelCatalog.Default,
            compressedAudioPath: null,
            generalTerms: [],
            generatedTopic: generatedTopic);
    }

    public static RecordingResult SaveTranscript(
        RecordingResult result,
        string transcript,
        TranscriptionModelOption model,
        string? compressedAudioPath,
        IReadOnlyList<string> generalTerms,
        string? generatedTopic = null)
    {
        if (!string.IsNullOrWhiteSpace(generatedTopic))
        {
            result = MoveToTopicDirectory(result, generatedTopic.Trim());
        }

        result = RenameRecordingForTopic(result);
        Directory.CreateDirectory(result.SessionDirectory);
        var path = Path.Combine(result.SessionDirectory, TranscriptFileNameForModel(result.Name, model.Id));
        var currentCompressedPath = Path.Combine(result.SessionDirectory, AudioCompressionService.CompressedFileName);
        var savedCompressedPath = File.Exists(currentCompressedPath)
            ? currentCompressedPath
            : compressedAudioPath;
        File.WriteAllText(
            path,
            BuildTranscriptMarkdown(result, transcript, model, savedCompressedPath, generalTerms),
            Encoding.UTF8);
        return result with { TranscriptPath = path };
    }

    public static RecordingResult SaveMinutes(RecordingResult result, string minutesMarkdown)
    {
        var topic = NeedsGeneratedTopic(result.Name)
            ? ExtractMarkdownTitle(minutesMarkdown) ?? result.Name
            : result.Name;
        result = MoveToTopicDirectory(result, topic);
        result = RenameRecordingForTopic(result);
        var path = Path.Combine(result.SessionDirectory, NotesFileNameForTopic(result.Name));
        File.WriteAllText(path, minutesMarkdown.Trim() + Environment.NewLine, Encoding.UTF8);
        return result with { MinutesPath = path };
    }

    public static string TranscriptFileNameForTopic(string topic) => $"transcription-{SlugOrUntitled(topic)}.md";

    public static string TranscriptFileNameForModel(string topic, string modelId) =>
        $"transcription-{SlugOrUntitled(modelId)}-{SlugOrUntitled(topic)}.md";

    public static string NotesFileNameForTopic(string topic) => $"notes-{SlugOrUntitled(topic)}.md";

    public static string RecordingFileNameForTopic(string topic) => $"recording-{SlugOrUntitled(topic)}.wav";

    public static bool NeedsGeneratedTopic(string? topic) =>
        string.IsNullOrWhiteSpace(topic) ||
        Slugify(topic).Equals("untitled-meeting", StringComparison.OrdinalIgnoreCase);

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
        var slug = builder.ToString().Trim('-');
        if (slug.Length <= MaxSlugLength)
        {
            return slug;
        }

        return slug[..MaxSlugLength].Trim('-');
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

    private static RecordingResult RenameRecordingForTopic(RecordingResult result)
    {
        if (string.IsNullOrWhiteSpace(result.MixedPath) || !File.Exists(result.MixedPath))
        {
            return result;
        }

        var target = Path.Combine(result.SessionDirectory, RecordingFileNameForTopic(result.Name));
        if (SamePath(target, result.MixedPath))
        {
            return result;
        }

        if (!File.Exists(target))
        {
            try
            {
                File.Move(result.MixedPath, target);
            }
            catch (IOException)
            {
                return result;
            }
            catch (UnauthorizedAccessException)
            {
                return result;
            }
        }

        return result with
        {
            MixedPath = target,
            Mixed = result.Mixed with { Path = target }
        };
    }

    private static string SlugOrUntitled(string value)
    {
        var slug = Slugify(value);
        return string.IsNullOrWhiteSpace(slug) ? "untitled-meeting" : slug;
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

        try
        {
            Directory.Move(result.SessionDirectory, target);
        }
        catch (IOException)
        {
            return result with { Name = topic };
        }
        catch (UnauthorizedAccessException)
        {
            return result with { Name = topic };
        }

        return result with
        {
            Name = topic,
            SessionDirectory = target,
            MicrophonePath = Repath(result.MicrophonePath, result.SessionDirectory, target),
            SystemPath = Repath(result.SystemPath, result.SessionDirectory, target),
            MixedPath = Repath(result.MixedPath, result.SessionDirectory, target),
            TranscriptPath = Repath(result.TranscriptPath, result.SessionDirectory, target),
            MinutesPath = Repath(result.MinutesPath, result.SessionDirectory, target),
            Microphone = Repath(result.Microphone, result.SessionDirectory, target),
            System = Repath(result.System, result.SessionDirectory, target),
            Mixed = Repath(result.Mixed, result.SessionDirectory, target)
        };
    }

    private static string BuildTranscriptMarkdown(
        RecordingResult result,
        string transcript,
        TranscriptionModelOption model,
        string? compressedAudioPath,
        IReadOnlyList<string> generalTerms)
    {
        var builder = new StringBuilder();
        builder.AppendLine($"# Speaker Transcript - {result.Name}");
        builder.AppendLine();
        builder.AppendLine($"Recorded: {result.StartedAt:yyyy-MM-dd HH:mm}");
        builder.AppendLine($"Duration: {result.Duration.TotalMinutes:F1} minutes");
        builder.AppendLine($"Transcription model: {model.DisplayName} ({model.Id})");
        builder.AppendLine($"Speaker diarization: {(model.IncludesSpeakerDiarization ? "included" : "not available in this model")}");
        if (model.Id == TranscriptionModelCatalog.GptTranscribeId)
        {
            builder.AppendLine(
                "Transcription pipeline: GPT Transcribe wording aligned to GPT-4o Transcribe Diarize speaker segments");
            builder.AppendLine("Authoritative wording: GPT Transcribe");
            builder.AppendLine("Speaker and timestamp source: GPT-4o Transcribe Diarize");
        }
        builder.AppendLine($"Transcription audio: {(string.IsNullOrWhiteSpace(compressedAudioPath) ? "compressed MP3" : Path.GetFileName(compressedAudioPath))}");
        if (generalTerms.Count > 0)
        {
            builder.AppendLine(model.SupportsGeneralTerms
                ? $"General terms applied: {string.Join(", ", generalTerms)}"
                : "General terms applied: model API does not support vocabulary hints");
        }
        builder.AppendLine($"Created: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
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

    private static TrackMetrics Repath(TrackMetrics metrics, string oldRoot, string newRoot) =>
        metrics with { Path = Repath(metrics.Path, oldRoot, newRoot) };

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
