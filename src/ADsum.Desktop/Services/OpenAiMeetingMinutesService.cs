using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace ADsum.Desktop.Services;

public sealed class OpenAiMeetingMinutesService
{
    private const int MaxTranscriptCharactersPerRequest = 60000;
    private const int MaxTopicTranscriptCharacters = 20000;
    private static readonly TimeSpan RequestTimeout = TimeSpan.FromMinutes(30);
    private static readonly HttpClient Client = new()
    {
        Timeout = RequestTimeout
    };

    public async Task<string> CreateMinutesAsync(
        string transcript,
        string? apiKey,
        string model,
        IProgress<string>? progress = null,
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

        if (transcript.Length > MaxTranscriptCharactersPerRequest)
        {
            return await CreateHierarchicalMinutesAsync(transcript, apiKey, model, progress, cancellationToken);
        }

        progress?.Report("Generating meeting minutes");
        return await CreateResponseAsync(
            BuildInstructions(),
            BuildInput(transcript),
            apiKey,
            model,
            cancellationToken);
    }

    public async Task<string> CreateTopicAsync(
        string transcript,
        string? apiKey,
        string model,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new InvalidOperationException("OpenAI key is not configured.");
        }

        if (string.IsNullOrWhiteSpace(transcript))
        {
            throw new InvalidOperationException("Generate a transcript before naming the meeting.");
        }

        progress?.Report("Naming meeting");
        var response = await CreateResponseAsync(
            BuildTopicInstructions(),
            BuildTopicInput(TopicTranscriptExcerpt(transcript)),
            apiKey,
            model,
            cancellationToken);
        return NormalizeTopicTitle(response);
    }

    private static async Task<string> CreateHierarchicalMinutesAsync(
        string transcript,
        string apiKey,
        string model,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        var chunks = SplitTranscript(transcript, MaxTranscriptCharactersPerRequest);
        var partials = new List<string>();
        for (var index = 0; index < chunks.Count; index++)
        {
            progress?.Report($"Summarizing transcript part {index + 1} of {chunks.Count}");
            partials.Add(await CreateResponseAsync(
                BuildPartialInstructions(),
                BuildPartialInput(chunks[index], index + 1, chunks.Count),
                apiKey,
                model,
                cancellationToken));
        }

        progress?.Report("Combining meeting minutes");
        return await CreateResponseAsync(
            BuildInstructions(),
            BuildFinalInput(partials),
            apiKey,
            model,
            cancellationToken);
    }

    private static async Task<string> CreateResponseAsync(
        string instructions,
        string input,
        string apiKey,
        string model,
        CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/responses");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        var payload = new
        {
            model,
            instructions,
            input
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
            throw new InvalidOperationException("OpenAI returned an empty response.");
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

    private static string BuildTopicInstructions()
    {
        return """
You name recorded meetings from their transcripts.
Identify the main subject of the meeting and return only a specific, descriptive title of 3 to 8 words.
Do not use Markdown, quotation marks, labels such as "Title:", or ending punctuation.
Do not invent a subject that is not supported by the transcript.
""";
    }

    private static string BuildTopicInput(string transcript)
    {
        return $"""
Create a short meeting title for this transcript.

Transcript:
{transcript}
""";
    }

    private static string TopicTranscriptExcerpt(string transcript)
    {
        if (transcript.Length <= MaxTopicTranscriptCharacters)
        {
            return transcript;
        }

        const int beginningCharacters = 14000;
        var endingCharacters = MaxTopicTranscriptCharacters - beginningCharacters;
        return transcript[..beginningCharacters] +
            "\n\n[Middle of transcript omitted for naming]\n\n" +
            transcript[^endingCharacters..];
    }

    private static string NormalizeTopicTitle(string response)
    {
        var title = response
            .Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.Trim())
            .FirstOrDefault(line => line.Length > 0)
            ?? "";

        foreach (var prefix in new[] { "Meeting topic:", "Meeting title:", "Topic:", "Title:" })
        {
            if (title.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                title = title[prefix.Length..].Trim();
                break;
            }
        }

        title = title.Trim(' ', '\t', '"', '\'', '`', '*', '#');
        title = title.TrimEnd('.', ',', ';', ':', '-', '\u2013', '\u2014');
        var words = title.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length > 8)
        {
            title = string.Join(' ', words.Take(8));
        }

        if (string.IsNullOrWhiteSpace(title))
        {
            throw new InvalidOperationException("OpenAI did not return a usable meeting title.");
        }

        return title;
    }

    private static string BuildInput(string transcript)
    {
        return $"""
Create meeting minutes for this transcript.

Transcript:
{transcript}
""";
    }

    private static string BuildPartialInstructions()
    {
        return """
You create practical partial meeting notes from one part of a longer speaker-labelled transcript.
Use only this transcript part as evidence. Do not invent decisions, owners, deadlines, or tasks.
Return clean Markdown only, with no code fences.

Required structure:
## Summary
One concise paragraph for this part.

## Important points discussed
- Bullet list.

## Tasks and next steps
- Bullet list. Include owner and due date when present. Use "Owner: Not specified" or "Due: Not specified" when absent.

## Decisions
- Bullet list of decisions. If none were made, write "- None identified."
""";
    }

    private static string BuildPartialInput(string transcript, int partNumber, int totalParts)
    {
        return $"""
Create partial meeting notes for transcript part {partNumber} of {totalParts}.

Transcript part:
{transcript}
""";
    }

    private static string BuildFinalInput(IReadOnlyList<string> partialMinutes)
    {
        var builder = new StringBuilder();
        builder.AppendLine("Create final meeting minutes by merging these partial notes from one long meeting.");
        builder.AppendLine("Remove duplicate points, preserve concrete tasks/decisions, and keep the required structure.");
        builder.AppendLine();
        for (var index = 0; index < partialMinutes.Count; index++)
        {
            builder.AppendLine($"Partial notes {index + 1}:");
            builder.AppendLine(partialMinutes[index]);
            builder.AppendLine();
        }
        return builder.ToString();
    }

    private static IReadOnlyList<string> SplitTranscript(string transcript, int maxCharacters)
    {
        var chunks = new List<string>();
        var current = new StringBuilder();
        foreach (var line in transcript.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None))
        {
            if (current.Length > 0 && current.Length + line.Length + 1 > maxCharacters)
            {
                chunks.Add(current.ToString());
                current.Clear();
            }

            current.AppendLine(line);
        }

        if (current.Length > 0)
        {
            chunks.Add(current.ToString());
        }

        return chunks;
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
