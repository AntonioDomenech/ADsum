using System.Globalization;
using System.Text;

namespace ADsum.Desktop.Services;

public static class MeetingTopicFallback
{
    private static readonly HashSet<string> IgnoredWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "about", "after", "again", "also", "and", "are", "because", "been", "before", "being", "can",
        "could", "did", "does", "doing", "each", "everyone", "for", "from", "going", "had", "has", "have",
        "hello", "how", "into", "its", "just", "know", "let", "meeting", "more", "not", "okay", "our",
        "out", "really", "recording", "should", "some", "speaker", "than", "thanks", "that", "the", "their",
        "then", "there", "these", "they", "thing", "think", "this", "those", "through", "today", "too",
        "transcript", "use", "very", "want", "was", "well", "were", "what", "when", "where", "which", "will",
        "with", "would", "yeah", "you", "your",
        "ahora", "aquí", "bien", "bueno", "como", "con", "cuando", "del", "donde", "entonces", "esta", "este",
        "esto", "gracias", "hacer", "hay", "hemos", "hola", "las", "los", "más", "muy", "nos", "para", "pero",
        "por", "porque", "que", "qué", "sin", "sobre", "son", "sus", "también", "tiene", "todos", "una", "uno",
        "vamos"
    };

    public static string FromTranscript(string transcript, DateTime startedAt)
    {
        var frequencies = new Dictionary<string, WordFrequency>(StringComparer.OrdinalIgnoreCase);
        var index = 0;
        foreach (var word in Words(transcript))
        {
            var key = word.ToLowerInvariant();
            if (!IsUsefulWord(key))
            {
                continue;
            }

            if (frequencies.TryGetValue(key, out var frequency))
            {
                frequencies[key] = frequency with { Count = frequency.Count + 1 };
            }
            else
            {
                frequencies[key] = new WordFrequency(word, 1, index);
            }
            index++;
        }

        var selected = frequencies.Values
            .OrderByDescending(word => word.Count)
            .ThenBy(word => word.FirstIndex)
            .Take(5)
            .OrderBy(word => word.FirstIndex)
            .Select(word => CultureInfo.CurrentCulture.TextInfo.ToTitleCase(word.Display.ToLowerInvariant()))
            .ToList();

        return selected.Count switch
        {
            0 => $"Recorded Meeting {startedAt:yyyy-MM-dd HHmm}",
            1 => $"Discussion About {selected[0]}",
            2 => $"{selected[0]} {selected[1]} Discussion",
            _ => string.Join(' ', selected)
        };
    }

    private static bool IsUsefulWord(string word)
    {
        if (IgnoredWords.Contains(word) || word.All(char.IsDigit) || word.Length > 40)
        {
            return false;
        }

        return word.Length >= 3 || word.Any(char.IsDigit) || word.Equals("ai", StringComparison.OrdinalIgnoreCase);
    }

    private static IEnumerable<string> Words(string text)
    {
        var current = new StringBuilder();
        foreach (var character in text)
        {
            if (char.IsLetterOrDigit(character))
            {
                current.Append(character);
                continue;
            }

            if (current.Length > 0)
            {
                yield return current.ToString();
                current.Clear();
            }
        }

        if (current.Length > 0)
        {
            yield return current.ToString();
        }
    }

    private sealed record WordFrequency(string Display, int Count, int FirstIndex);
}
