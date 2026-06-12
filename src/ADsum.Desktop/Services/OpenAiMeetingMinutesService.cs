using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace ADsum.Desktop.Services;

public sealed class OpenAiMeetingMinutesService
{
    private static readonly HttpClient Client = new();

    public async Task<string> CreateMinutesAsync(
        string transcript,
        string? apiKey,
        string model,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new InvalidOperationException("OpenAI key is not configured.");
        }

        if (string.IsNullOrWhiteSpace(transcript))
        {
            throw new InvalidOperationException("Generate a transcript before creating meeting minutes.");
        }

        using var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/responses");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        var payload = new
        {
            model,
            instructions = BuildInstructions(),
            input = BuildInput(transcript)
        };
        request.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

        using var response = await Client.SendAsync(request, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException($"OpenAI meeting minutes failed: {(int)response.StatusCode} {response.ReasonPhrase}\n{body}");
        }

        var text = ExtractOutputText(body);
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new InvalidOperationException("OpenAI returned an empty meeting-minutes response.");
        }
        return text.Trim();
    }

    private static string BuildInstructions()
    {
        return """
You create practical meeting minutes from speaker-labelled transcripts.
Use only the transcript as evidence. Do not invent decisions, owners, deadlines, or tasks.
If something is unclear, mark it as "Not specified" rather than guessing.
Return clean Markdown only, with no code fences.

Required structure:
# <short topic title, 3-8 words, suitable for a folder name>

## Summary
One concise paragraph describing what the meeting was about.

## Important points discussed
- Bullet list of the most important points.

## Tasks and next steps
- Bullet list. Include owner and due date when present. Use "Owner: Not specified" or "Due: Not specified" when absent.

## Decisions
- Bullet list of decisions. If none were made, write "- None identified."
""";
    }

    private static string BuildInput(string transcript)
    {
        return $"""
Create meeting minutes for this transcript.

Transcript:
{transcript}
""";
    }

    private static string ExtractOutputText(string body)
    {
        using var document = JsonDocument.Parse(body);
        var root = document.RootElement;
        if (root.TryGetProperty("output_text", out var outputText) && outputText.ValueKind == JsonValueKind.String)
        {
            return outputText.GetString() ?? "";
        }

        var builder = new StringBuilder();
        if (root.TryGetProperty("output", out var output) && output.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in output.EnumerateArray())
            {
                if (!item.TryGetProperty("content", out var content) || content.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                foreach (var contentItem in content.EnumerateArray())
                {
                    if (contentItem.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String)
                    {
                        builder.AppendLine(text.GetString());
                    }
                }
            }
        }

        return builder.ToString();
    }
}
