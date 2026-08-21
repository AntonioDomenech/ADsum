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
    private const int ChunkMinutes = 20;
    private static readonly TimeSpan RequestTimeout = TimeSpan.FromMinutes(30);
    private static readonly HttpClient Client = new() { Timeout = RequestTimeout };

    public Task<string> TranscribeAsync(
        string audioPath,
        string? apiKey,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default) =>
        TranscribeAsync(
            audioPath,
            apiKey,
            TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId,
            Array.Empty<string>(),
            progress,
            cancellationToken);

    public async Task<string> TranscribeAsync(
        string audioPath,
        string? apiKey,
        string modelId,
        IReadOnlyList<string> generalTerms,
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

        var model = TranscriptionModelCatalog.Resolve(modelId);
        if (!model.RequiresOpenAiKey)
        {
            throw new InvalidOperationException($"{model.DisplayName} is not an OpenAI transcription model.");
        }

        var normalizedTerms = NormalizeTerms(generalTerms);
        if (normalizedTerms.Count > 0 && !model.SupportsGeneralTerms)
        {
            progress?.Report("This diarization model does not accept vocabulary hints; continuing with its supported API fields");
        }

        if (new FileInfo(audioPath).Length <= MaxUploadBytes)
        {
            try
            {
                var single = await TranscribeSelectedSingleFileAsync(
                        audioPath,
                        apiKey,
                        model,
                        normalizedTerms,
                        TimeSpan.Zero,
                        progress,
                        cancellationToken)
                    .ConfigureAwait(false);
                return FormatDiarizedTranscript(single, wasChunked: false);
            }
            catch (InvalidOperationException ex) when (
                (model.Id == TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId ||
                 model.Id == TranscriptionModelCatalog.GptTranscribeId) &&
                IsRecoverableFullRecordingError(ex))
            {
                progress?.Report("The full recording exceeded the speaker-model limit; splitting the compressed MP3");
            }
        }

        var tempDirectory = Path.Combine(
            Path.GetTempPath(),
            "ADsum",
            "OpenAiTranscriptionChunks",
            Guid.NewGuid().ToString("N"));
        try
        {
            progress?.Report("Splitting compressed MP3 for upload");
            var chunks = CreateCompressedUploadChunks(audioPath, tempDirectory, cancellationToken);
            if (chunks.Count == 0)
            {
                throw new InvalidDataException("The compressed MP3 did not contain readable audio.");
            }

            return await TranscribeDiarizedChunksAsync(
                    chunks, apiKey, model, normalizedTerms, progress, cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            TryDeleteDirectory(tempDirectory);
        }
    }

    private static async Task<string> TranscribeDiarizedChunksAsync(
        IReadOnlyList<UploadChunk> chunks,
        string apiKey,
        TranscriptionModelOption model,
        IReadOnlyList<string> generalTerms,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        var segments = new List<DiarizedTextSegment>();
        var fallbackText = new StringBuilder();
        var notices = new List<string>();
        for (var index = 0; index < chunks.Count; index++)
        {
            var chunk = chunks[index];
            progress?.Report($"Transcribing and diarizing MP3 part {index + 1} of {chunks.Count}");
            var result = await TranscribeSelectedSingleFileAsync(
                    chunk.Path,
                    apiKey,
                    model,
                    generalTerms,
                    chunk.Offset,
                    progress,
                    cancellationToken)
                .ConfigureAwait(false);
            segments.AddRange(result.Segments.Select(segment =>
                segment with { Speaker = $"{chunk.Index}:{segment.Speaker}" }));
            if (!string.IsNullOrWhiteSpace(result.Text))
            {
                fallbackText.AppendLine(result.Text.Trim());
                fallbackText.AppendLine();
            }
            if (!string.IsNullOrWhiteSpace(result.Notice))
            {
                notices.Add($"Part {chunk.Index}: {result.Notice}");
            }
        }

        return FormatDiarizedTranscript(
            new TranscriptionResponse(
                segments,
                fallbackText.ToString(),
                notices.Count == 0 ? null : string.Join(" ", notices)),
            wasChunked: true);
    }

    private static async Task<TranscriptionResponse> TranscribeSelectedSingleFileAsync(
        string audioPath,
        string apiKey,
        TranscriptionModelOption selectedModel,
        IReadOnlyList<string> generalTerms,
        TimeSpan offset,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        if (selectedModel.Id != TranscriptionModelCatalog.GptTranscribeId)
        {
            return await TranscribeSingleFileAsync(
                    audioPath,
                    apiKey,
                    selectedModel,
                    generalTerms,
                    offset,
                    cancellationToken)
                .ConfigureAwait(false);
        }

        progress?.Report("Running GPT Transcribe wording and GPT-4o speaker labeling in parallel");
        var wordingModel = TranscriptionModelCatalog.Resolve(TranscriptionModelCatalog.GptTranscribeId);
        var speakerModel = TranscriptionModelCatalog.Resolve(TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId);
        var wordingTask = TranscribeSingleFileAsync(
            audioPath,
            apiKey,
            wordingModel,
            generalTerms,
            offset,
            cancellationToken);
        var speakerTask = TranscribeSingleFileAsync(
            audioPath,
            apiKey,
            speakerModel,
            Array.Empty<string>(),
            offset,
            cancellationToken);
        await Task.WhenAll(wordingTask, speakerTask).ConfigureAwait(false);

        var wording = await wordingTask.ConfigureAwait(false);
        var speakerTranscript = await speakerTask.ConfigureAwait(false);
        var alignment = DiarizedTranscriptAligner.Align(wording.Text, speakerTranscript.Segments);
        var notice = alignment.UsedProportionalFallback
            ? $"GPT Transcribe remained the authoritative wording. {alignment.Reason}"
            : null;
        return new TranscriptionResponse(alignment.Segments, wording.Text, notice);
    }

    private static async Task<TranscriptionResponse> TranscribeSingleFileAsync(
        string audioPath,
        string apiKey,
        TranscriptionModelOption model,
        IReadOnlyList<string> generalTerms,
        TimeSpan offset,
        CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/audio/transcriptions");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        await using var stream = File.OpenRead(audioPath);
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(model.Id), "model");
        if (model.Id == TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId)
        {
            content.Add(new StringContent("diarized_json"), "response_format");
            content.Add(new StringContent("auto"), "chunking_strategy");
        }
        else if (model.Id == TranscriptionModelCatalog.GptTranscribeId && generalTerms.Count > 0)
        {
            content.Add(
                new StringContent($"Relevant names and specialist terms may include: {string.Join(", ", generalTerms)}."),
                "prompt");
            foreach (var term in generalTerms)
            {
                content.Add(new StringContent(term), "keywords[]");
            }
        }

        using var fileContent = new StreamContent(stream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(ContentTypeFor(audioPath));
        content.Add(fileContent, "file", Path.GetFileName(audioPath));
        request.Content = content;

        using var response = await Client.SendAsync(request, cancellationToken).ConfigureAwait(false);
        var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException(
                $"OpenAI transcription failed with {model.DisplayName}: {(int)response.StatusCode} {response.ReasonPhrase}\n{body}");
        }

        return model.Id == TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId
            ? ParseDiarizedResponse(body, offset)
            : ParsePlainResponse(body);
    }

    private static TranscriptionResponse ParsePlainResponse(string body)
    {
        using var document = JsonDocument.Parse(body);
        var text = document.RootElement.TryGetProperty("text", out var textElement)
            ? textElement.GetString() ?? ""
            : "";
        return new TranscriptionResponse([], text, null);
    }

    private static TranscriptionResponse ParseDiarizedResponse(string body, TimeSpan offset)
    {
        using var document = JsonDocument.Parse(body);
        var root = document.RootElement;
        var text = root.TryGetProperty("text", out var textElement)
            ? textElement.GetString() ?? ""
            : "";
        var segments = new List<DiarizedTextSegment>();
        if (root.TryGetProperty("segments", out var segmentElements) && segmentElements.ValueKind == JsonValueKind.Array)
        {
            foreach (var segment in segmentElements.EnumerateArray())
            {
                var segmentText = ReadString(segment, "text");
                if (string.IsNullOrWhiteSpace(segmentText))
                {
                    continue;
                }

                segments.Add(new DiarizedTextSegment(
                    offset + TimeSpan.FromSeconds(ReadDouble(segment, "start")),
                    offset + TimeSpan.FromSeconds(ReadDouble(segment, "end")),
                    ReadString(segment, "speaker"),
                    segmentText.Trim()));
            }
        }

        return new TranscriptionResponse(segments, text, null);
    }

    private static string FormatDiarizedTranscript(TranscriptionResponse transcript, bool wasChunked)
    {
        if (transcript.Segments.Count == 0)
        {
            if (string.IsNullOrWhiteSpace(transcript.Text))
            {
                return "";
            }

            throw new InvalidDataException(
                "OpenAI returned transcript text without speaker segments. ADsum did not save an un-diarized result.");
        }

        var labels = new SpeakerLabeler();
        var output = new StringBuilder();
        if (!string.IsNullOrWhiteSpace(transcript.Notice))
        {
            output.AppendLine($"Note: {transcript.Notice}");
            output.AppendLine();
        }
        if (wasChunked)
        {
            output.AppendLine("Note: the compressed MP3 was split for long-meeting reliability. Speaker labels may reset between parts.");
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

    private static IReadOnlyList<UploadChunk> CreateCompressedUploadChunks(
        string audioPath,
        string directory,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(directory);
        var chunks = new List<UploadChunk>();
        using var reader = new AudioFileReader(audioPath);
        var samplesPerChunk = checked((long)reader.WaveFormat.SampleRate * reader.WaveFormat.Channels * 60 * ChunkMinutes);
        var buffer = new float[Math.Max(reader.WaveFormat.SampleRate * reader.WaveFormat.Channels, 4096)];
        long totalSamples = 0;
        var index = 1;

        while (reader.Position < reader.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var wavePath = Path.Combine(directory, $"part-{index:0000}.tmp.wav");
            var mp3Path = Path.Combine(directory, $"part-{index:0000}.mp3");
            long writtenSamples = 0;
            using (var writer = new WaveFileWriter(wavePath, reader.WaveFormat))
            {
                while (writtenSamples < samplesPerChunk)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var requested = (int)Math.Min(buffer.Length, samplesPerChunk - writtenSamples);
                    var read = reader.Read(buffer, 0, requested);
                    if (read <= 0)
                    {
                        break;
                    }

                    writer.WriteSamples(buffer, 0, read);
                    writtenSamples += read;
                }
            }

            if (writtenSamples <= 0)
            {
                TryDeleteFile(wavePath);
                break;
            }

            using (var chunkReader = new AudioFileReader(wavePath))
            {
                MediaFoundationApi.Startup();
                MediaFoundationEncoder.EncodeToMp3(chunkReader, mp3Path, AudioCompressionService.SpeechBitRate);
            }
            TryDeleteFile(wavePath);

            if (!File.Exists(mp3Path) || new FileInfo(mp3Path).Length <= 0 || new FileInfo(mp3Path).Length > MaxUploadBytes)
            {
                throw new InvalidDataException($"Compressed MP3 part {index} is not a valid upload size.");
            }

            var offsetSeconds = (double)totalSamples /
                Math.Max(1, reader.WaveFormat.SampleRate * reader.WaveFormat.Channels);
            chunks.Add(new UploadChunk(index, mp3Path, TimeSpan.FromSeconds(offsetSeconds)));
            totalSamples += writtenSamples;
            index++;
        }

        return chunks;
    }

    private static IReadOnlyList<string> NormalizeTerms(IEnumerable<string> values) => values
        .Select(value => value.Trim())
        .Where(value => value.Length > 0)
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToList();

    private static bool IsRecoverableFullRecordingError(Exception ex)
    {
        var message = ex.Message;
        return (message.Contains("\"param\": \"file\"", StringComparison.OrdinalIgnoreCase) &&
                message.Contains("\"code\": \"invalid_value\"", StringComparison.OrdinalIgnoreCase)) ||
               (message.Contains("audio duration", StringComparison.OrdinalIgnoreCase) &&
                message.Contains("longer than", StringComparison.OrdinalIgnoreCase));
    }

    private static string FormatTimestamp(TimeSpan value) => value.TotalHours >= 1
        ? value.ToString(@"h\:mm\:ss")
        : value.ToString(@"m\:ss");

    private static string ReadString(JsonElement element, string propertyName) =>
        element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? ""
            : "";

    private static double ReadDouble(JsonElement element, string propertyName) =>
        element.TryGetProperty(propertyName, out var value) && value.TryGetDouble(out var number)
            ? number
            : 0;

    private static string ContentTypeFor(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".mp3" => "audio/mpeg",
        ".m4a" or ".mp4" => "audio/mp4",
        ".ogg" => "audio/ogg",
        ".webm" => "audio/webm",
        _ => "audio/wav"
    };

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
            // Temporary upload cleanup is best effort.
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
            // Temporary upload cleanup is best effort.
        }
    }

    private sealed record TranscriptionResponse(
        IReadOnlyList<DiarizedTextSegment> Segments,
        string Text,
        string? Notice);
    private sealed record UploadChunk(int Index, string Path, TimeSpan Offset);

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
