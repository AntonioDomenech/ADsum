using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using NAudio.Wave;

namespace ADsum.Desktop.Services;

public sealed class OpenAiTranscriptionService
{
    private const long MaxUploadBytes = 24L * 1024 * 1024;
    private const string DiarizationModel = "gpt-4o-transcribe-diarize";
    private static readonly HttpClient Client = new();

    public async Task<string> TranscribeAsync(string audioPath, string? apiKey, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new InvalidOperationException("OpenAI key is not configured.");
        }

        if (!File.Exists(audioPath))
        {
            throw new FileNotFoundException("Audio file was not found.", audioPath);
        }

        if (new FileInfo(audioPath).Length <= MaxUploadBytes)
        {
            var transcript = await TranscribeSingleFileAsync(audioPath, apiKey, TimeSpan.Zero, cancellationToken);
            return FormatDiarizedTranscript(transcript, wasChunked: false);
        }

        var tempDirectory = Path.Combine(Path.GetTempPath(), "ADsum", "TranscriptionChunks", Guid.NewGuid().ToString("N"));
        try
        {
            var chunks = CreateUploadChunks(audioPath, tempDirectory);
            var segments = new List<DiarizedSegment>();
            var fallbackText = new StringBuilder();
            foreach (var chunk in chunks)
            {
                var transcript = await TranscribeSingleFileAsync(chunk.Path, apiKey, chunk.Offset, cancellationToken);
                segments.AddRange(transcript.Segments);
                if (!string.IsNullOrWhiteSpace(transcript.Text))
                {
                    fallbackText.AppendLine(transcript.Text.Trim());
                    fallbackText.AppendLine();
                }
            }
            return FormatDiarizedTranscript(new DiarizedTranscript(segments, fallbackText.ToString()), wasChunked: true);
        }
        finally
        {
            TryDeleteDirectory(tempDirectory);
        }
    }

    private static async Task<DiarizedTranscript> TranscribeSingleFileAsync(
        string audioPath,
        string apiKey,
        TimeSpan offset,
        CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/audio/transcriptions");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        await using var stream = File.OpenRead(audioPath);
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(DiarizationModel), "model");
        content.Add(new StringContent("diarized_json"), "response_format");
        content.Add(new StringContent("auto"), "chunking_strategy");
        using var fileContent = new StreamContent(stream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
        content.Add(fileContent, "file", Path.GetFileName(audioPath));
        request.Content = content;

        using var response = await Client.SendAsync(request, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException($"OpenAI transcription failed: {(int)response.StatusCode} {response.ReasonPhrase}\n{body}");
        }

        return ParseDiarizedResponse(body, offset);
    }

    private static DiarizedTranscript ParseDiarizedResponse(string body, TimeSpan offset)
    {
        using var document = JsonDocument.Parse(body);
        var root = document.RootElement;
        var text = root.TryGetProperty("text", out var textElement)
            ? textElement.GetString() ?? ""
            : "";

        var segments = new List<DiarizedSegment>();
        if (root.TryGetProperty("segments", out var segmentElements) && segmentElements.ValueKind == JsonValueKind.Array)
        {
            foreach (var segment in segmentElements.EnumerateArray())
            {
                var segmentText = ReadString(segment, "text");
                if (string.IsNullOrWhiteSpace(segmentText))
                {
                    continue;
                }

                var speaker = ReadString(segment, "speaker");
                var start = offset + TimeSpan.FromSeconds(ReadDouble(segment, "start"));
                var end = offset + TimeSpan.FromSeconds(ReadDouble(segment, "end"));
                segments.Add(new DiarizedSegment(start, end, speaker, segmentText.Trim()));
            }
        }

        return new DiarizedTranscript(segments, text);
    }

    private static string FormatDiarizedTranscript(DiarizedTranscript transcript, bool wasChunked)
    {
        if (transcript.Segments.Count == 0)
        {
            return string.IsNullOrWhiteSpace(transcript.Text)
                ? ""
                : transcript.Text.Trim();
        }

        var labels = new SpeakerLabeler();
        var output = new StringBuilder();
        if (wasChunked)
        {
            output.AppendLine("Note: this recording exceeded the upload-size limit and was split before transcription. Speaker labels may reset between upload chunks.");
            output.AppendLine();
        }

        foreach (var segment in transcript.Segments.OrderBy(segment => segment.Start))
        {
            output
                .Append(FormatTimestamp(segment.Start))
                .Append(" - ")
                .Append(FormatTimestamp(segment.End))
                .Append("  ")
                .Append(labels.DisplayName(segment.Speaker))
                .Append(": ")
                .AppendLine(segment.Text);
        }

        return output.ToString().Trim();
    }

    private static string FormatTimestamp(TimeSpan value)
    {
        return value.TotalHours >= 1
            ? value.ToString(@"h\:mm\:ss")
            : value.ToString(@"m\:ss");
    }

    private static string ReadString(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? ""
            : "";
    }

    private static double ReadDouble(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.TryGetDouble(out var number)
            ? number
            : 0;
    }

    private static IReadOnlyList<UploadChunk> CreateUploadChunks(string audioPath, string directory)
    {
        Directory.CreateDirectory(directory);
        using var reader = new WaveFileReader(audioPath);
        var blockAlign = Math.Max(1, reader.WaveFormat.BlockAlign);
        var maxDataBytes = MaxUploadBytes - 4096;
        maxDataBytes -= maxDataBytes % blockAlign;
        var bufferSize = (int)Math.Min(maxDataBytes, 1024 * 1024);
        bufferSize -= bufferSize % blockAlign;
        if (bufferSize <= 0)
        {
            bufferSize = blockAlign;
        }

        var buffer = new byte[bufferSize];
        var chunks = new List<UploadChunk>();
        var index = 1;
        long sourceBytesRead = 0;
        while (sourceBytesRead < reader.Length)
        {
            var chunkPath = Path.Combine(directory, $"chunk-{index:0000}.wav");
            var offset = TimeSpan.FromSeconds((double)sourceBytesRead / Math.Max(1, reader.WaveFormat.AverageBytesPerSecond));
            using (var writer = new WaveFileWriter(chunkPath, reader.WaveFormat))
            {
                long written = 0;
                while (written < maxDataBytes)
                {
                    var requested = (int)Math.Min(buffer.Length, maxDataBytes - written);
                    requested -= requested % blockAlign;
                    if (requested <= 0)
                    {
                        break;
                    }

                    var read = reader.Read(buffer, 0, requested);
                    if (read <= 0)
                    {
                        break;
                    }
                    writer.Write(buffer, 0, read);
                    written += read;
                    sourceBytesRead += read;
                }
            }
            chunks.Add(new UploadChunk(chunkPath, offset));
            index++;
        }
        return chunks;
    }

    private static void TryDeleteDirectory(string directory)
    {
        try
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
        catch
        {
            // Temporary chunk cleanup should not hide a successful transcription.
        }
    }

    private sealed record DiarizedTranscript(IReadOnlyList<DiarizedSegment> Segments, string Text);

    private sealed record DiarizedSegment(TimeSpan Start, TimeSpan End, string Speaker, string Text);

    private sealed record UploadChunk(string Path, TimeSpan Offset);

    private sealed class SpeakerLabeler
    {
        private readonly Dictionary<string, string> _labels = new(StringComparer.OrdinalIgnoreCase);
        private int _nextLabelIndex;

        public string DisplayName(string rawSpeaker)
        {
            var key = string.IsNullOrWhiteSpace(rawSpeaker) ? "unknown" : rawSpeaker.Trim();
            if (!_labels.TryGetValue(key, out var label))
            {
                label = $"Speaker {NextLabel()}";
                _labels[key] = label;
            }
            return label;
        }

        private string NextLabel()
        {
            var index = _nextLabelIndex++;
            var label = new StringBuilder();
            do
            {
                label.Insert(0, (char)('A' + (index % 26)));
                index = (index / 26) - 1;
            }
            while (index >= 0);
            return label.ToString();
        }
    }
}
