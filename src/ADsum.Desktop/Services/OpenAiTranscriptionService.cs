using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using NAudio.Wave;

namespace ADsum.Desktop.Services;

public sealed class OpenAiTranscriptionService
{
    private const long MaxUploadBytes = 24L * 1024 * 1024;
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
            return await TranscribeSingleFileAsync(audioPath, apiKey, cancellationToken);
        }

        var tempDirectory = Path.Combine(Path.GetTempPath(), "ADsum", "TranscriptionChunks", Guid.NewGuid().ToString("N"));
        try
        {
            var chunks = CreateUploadChunks(audioPath, tempDirectory);
            var transcripts = new List<string>();
            foreach (var chunk in chunks)
            {
                var text = await TranscribeSingleFileAsync(chunk, apiKey, cancellationToken);
                if (!string.IsNullOrWhiteSpace(text))
                {
                    transcripts.Add(text.Trim());
                }
            }
            return string.Join("\n\n", transcripts);
        }
        finally
        {
            TryDeleteDirectory(tempDirectory);
        }
    }

    private static async Task<string> TranscribeSingleFileAsync(string audioPath, string apiKey, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/audio/transcriptions");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        await using var stream = File.OpenRead(audioPath);
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent("gpt-4o-mini-transcribe"), "model");
        content.Add(new StringContent("json"), "response_format");
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

        using var document = JsonDocument.Parse(body);
        return document.RootElement.TryGetProperty("text", out var text)
            ? text.GetString() ?? ""
            : "";
    }

    private static IReadOnlyList<string> CreateUploadChunks(string audioPath, string directory)
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
        var chunks = new List<string>();
        var index = 1;
        while (reader.Position < reader.Length)
        {
            var chunkPath = Path.Combine(directory, $"chunk-{index:0000}.wav");
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
                }
            }
            chunks.Add(chunkPath);
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
}
