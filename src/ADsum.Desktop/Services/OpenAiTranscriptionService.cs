using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using NAudio.MediaFoundation;
using NAudio.Wave;

namespace ADsum.Desktop.Services;

public sealed class OpenAiTranscriptionService
{
    private const long MaxUploadBytes = 24L * 1024 * 1024;
    private const long UploadSafetyBytes = 1024 * 1024;
    private const int MaximumSpeechBitRate = 32000;
    private const int MinimumSpeechBitRate = 8000;
    private static readonly TimeSpan MaximumDiarizationDuration = TimeSpan.FromSeconds(1400);
    private static readonly TimeSpan ChunkDurationSafetyMargin = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan RequestTimeout = TimeSpan.FromMinutes(30);
    private const string DiarizationModel = "gpt-4o-transcribe-diarize";
    private static readonly HttpClient Client = new()
    {
        Timeout = RequestTimeout
    };

    public async Task<string> TranscribeAsync(
        string audioPath,
        string? apiKey,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new InvalidOperationException("OpenAI key is not configured.");
        }

        if (!File.Exists(audioPath))
        {
            throw new FileNotFoundException("Audio file was not found.", audioPath);
        }

        var duration = AudioDuration(audioPath);
        if (CanUploadSingleFile(audioPath, duration))
        {
            progress?.Report("Diarizing audio");
            var transcript = await TranscribeSingleFileAsync(audioPath, apiKey, TimeSpan.Zero, cancellationToken);
            return FormatDiarizedTranscript(transcript, wasChunked: false);
        }

        var tempDirectory = Path.Combine(Path.GetTempPath(), "ADsum", "TranscriptionChunks", Guid.NewGuid().ToString("N"));
        try
        {
            if (duration <= MaximumDiarizationDuration)
            {
                progress?.Report("Compressing recording for upload");
                foreach (var compressedUpload in CreateCompressedUploadCopies(audioPath, tempDirectory, duration))
                {
                    try
                    {
                        progress?.Report($"Diarizing full recording ({compressedUpload.Label})");
                        var transcript = await TranscribeSingleFileAsync(compressedUpload.Path, apiKey, TimeSpan.Zero, cancellationToken);
                        return FormatDiarizedTranscript(transcript, wasChunked: false);
                    }
                    catch (InvalidOperationException ex) when (IsRecoverableFullRecordingError(ex))
                    {
                        progress?.Report($"Full recording upload rejected ({compressedUpload.Label}); trying another format");
                        TryDeleteFile(compressedUpload.Path);
                    }
                }
            }
            else
            {
                progress?.Report("Recording is longer than the diarization limit; splitting for upload");
            }

            progress?.Report("Splitting recording for upload");
            var chunks = CreateUploadChunks(audioPath, tempDirectory);
            var segments = new List<DiarizedSegment>();
            var fallbackText = new StringBuilder();
            var chunkNumber = 1;
            foreach (var chunk in chunks)
            {
                progress?.Report($"Diarizing chunk {chunkNumber} of {chunks.Count}");
                var transcript = await TranscribeSingleFileAsync(chunk.Path, apiKey, chunk.Offset, cancellationToken);
                transcript = transcript with
                {
                    Segments = transcript.Segments
                        .Select(segment => segment with { Speaker = $"{chunk.Index}:{segment.Speaker}" })
                        .ToList()
                };
                segments.AddRange(transcript.Segments);
                if (!string.IsNullOrWhiteSpace(transcript.Text))
                {
                    fallbackText.AppendLine(transcript.Text.Trim());
                    fallbackText.AppendLine();
                }
                chunkNumber++;
            }
            progress?.Report("Formatting transcript");
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
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(ContentTypeFor(audioPath));
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
            output.AppendLine("Note: this recording was split into smaller transcription chunks for long-meeting reliability. Speaker labels may reset between chunks.");
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
        var durationLimit = MaximumDiarizationDuration - ChunkDurationSafetyMargin;
        var maxDurationBytes = (long)Math.Floor(durationLimit.TotalSeconds * Math.Max(1, reader.WaveFormat.AverageBytesPerSecond));
        var maxDataBytes = Math.Min(MaxUploadBytes - 4096, maxDurationBytes);
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
            chunks.Add(new UploadChunk(index, chunkPath, offset));
            index++;
        }
        return chunks;
    }

    private static bool CanUploadSingleFile(string audioPath, TimeSpan duration)
    {
        return duration <= MaximumDiarizationDuration
            && new FileInfo(audioPath).Length <= MaxUploadBytes;
    }

    private static IEnumerable<CompressedUpload> CreateCompressedUploadCopies(string audioPath, string directory, TimeSpan duration)
    {
        Directory.CreateDirectory(directory);
        if (duration <= TimeSpan.Zero)
        {
            yield break;
        }

        var bitRates = CandidateBitRates(duration);
        foreach (var bitRate in bitRates)
        {
            foreach (var encoder in new[] { CompressedEncoder.Mp3, CompressedEncoder.Aac })
            {
                var extension = encoder == CompressedEncoder.Mp3 ? ".mp3" : ".m4a";
                var path = Path.Combine(directory, $"upload-{bitRate}-{encoder.ToString().ToLowerInvariant()}{extension}");
                CompressedUpload? upload = null;
                try
                {
                    EncodeCompressed(audioPath, path, bitRate, encoder);
                    if (File.Exists(path) && new FileInfo(path).Length <= MaxUploadBytes)
                    {
                        upload = new CompressedUpload(path, bitRate, encoder);
                    }
                }
                catch
                {
                    TryDeleteFile(path);
                }

                if (upload is not null)
                {
                    yield return upload;
                }
            }
        }
    }

    private static bool IsRecoverableFullRecordingError(Exception ex)
    {
        return IsInvalidAudioUploadError(ex) || IsDiarizationDurationLimitError(ex);
    }

    private static bool IsInvalidAudioUploadError(Exception ex)
    {
        var message = ex.Message;
        return message.Contains("\"param\": \"file\"", StringComparison.OrdinalIgnoreCase)
            && message.Contains("\"code\": \"invalid_value\"", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsDiarizationDurationLimitError(Exception ex)
    {
        var message = ex.Message;
        return message.Contains("audio duration", StringComparison.OrdinalIgnoreCase)
            && message.Contains("longer than", StringComparison.OrdinalIgnoreCase)
            && message.Contains("1400 seconds", StringComparison.OrdinalIgnoreCase)
            && message.Contains("\"code\": \"invalid_value\"", StringComparison.OrdinalIgnoreCase);
    }

    private static IReadOnlyList<int> CandidateBitRates(TimeSpan duration)
    {
        var maxBytes = Math.Max(1, MaxUploadBytes - UploadSafetyBytes);
        var highestBitRateThatFits = (int)Math.Floor(maxBytes * 8.0 / Math.Max(1, duration.TotalSeconds));
        var preferred = Math.Clamp(highestBitRateThatFits, MinimumSpeechBitRate, MaximumSpeechBitRate);
        return new[] { preferred, 32000, 24000, 16000, 12000, 8000 }
            .Where(bitRate => bitRate <= highestBitRateThatFits || bitRate == preferred)
            .Where(bitRate => bitRate >= MinimumSpeechBitRate)
            .Distinct()
            .OrderByDescending(bitRate => bitRate)
            .ToList();
    }

    private static void EncodeCompressed(string inputPath, string outputPath, int bitRate, CompressedEncoder encoder)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        using var reader = new WaveFileReader(inputPath);
        MediaFoundationApi.Startup();
        if (encoder == CompressedEncoder.Mp3)
        {
            MediaFoundationEncoder.EncodeToMp3(reader, outputPath, bitRate);
            return;
        }

        MediaFoundationEncoder.EncodeToAac(reader, outputPath, bitRate);
    }

    private static TimeSpan AudioDuration(string audioPath)
    {
        using var reader = new WaveFileReader(audioPath);
        return reader.TotalTime;
    }

    private static string ContentTypeFor(string path)
    {
        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".mp3" => "audio/mpeg",
            ".m4a" => "audio/mp4",
            ".mp4" => "audio/mp4",
            ".ogg" => "audio/ogg",
            ".webm" => "audio/webm",
            _ => "audio/wav"
        };
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Temporary upload cleanup should not hide the underlying transcription path.
        }
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

    private sealed record UploadChunk(int Index, string Path, TimeSpan Offset);

    private sealed record CompressedUpload(string Path, int BitRate, CompressedEncoder Encoder)
    {
        public string Label => $"{Encoder.ToString().ToLowerInvariant()} {BitRate / 1000} kbps";
    }

    private enum CompressedEncoder
    {
        Mp3,
        Aac
    }

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
