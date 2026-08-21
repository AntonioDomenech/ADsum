using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace ADsum.Desktop.Services;

public sealed record DiarizedTextSegment(
    TimeSpan Start,
    TimeSpan End,
    string Speaker,
    string Text);

public sealed record DiarizedAlignmentResult(
    IReadOnlyList<DiarizedTextSegment> Segments,
    bool UsedAccurateText,
    bool UsedProportionalFallback,
    double ExactMatchRatio,
    int EmptySegments,
    string Reason);

public static partial class DiarizedTranscriptAligner
{
    private const int AnchorLookahead = 48;

    public static DiarizedAlignmentResult Align(
        string accurateText,
        IReadOnlyList<DiarizedTextSegment> diarizedSegments)
    {
        ArgumentNullException.ThrowIfNull(accurateText);
        ArgumentNullException.ThrowIfNull(diarizedSegments);

        var accurateTokens = Tokenize(accurateText);
        var diarizedTokens = diarizedSegments
            .SelectMany((segment, segmentIndex) =>
                Tokenize(segment.Text).Select(token => new SpeakerToken(token, segmentIndex)))
            .ToList();

        if (diarizedSegments.Count == 0 || diarizedTokens.Count == 0)
        {
            throw new InvalidDataException("The speaker-labeling pass returned no usable speaker segments.");
        }

        if (accurateTokens.Count == 0)
        {
            throw new InvalidDataException("GPT Transcribe returned no usable transcript words.");
        }

        var accurateToDiarized = Enumerable.Repeat(-1, accurateTokens.Count).ToArray();
        var accurateIndex = 0;
        var diarizedIndex = 0;
        var exactMatches = 0;

        while (accurateIndex < accurateTokens.Count && diarizedIndex < diarizedTokens.Count)
        {
            if (SameWord(accurateTokens[accurateIndex], diarizedTokens[diarizedIndex].Token))
            {
                accurateToDiarized[accurateIndex] = diarizedIndex;
                accurateIndex++;
                diarizedIndex++;
                exactMatches++;
                continue;
            }

            var anchor = FindNextAnchor(
                accurateTokens,
                accurateIndex,
                diarizedTokens,
                diarizedIndex);
            if (anchor is null)
            {
                accurateToDiarized[accurateIndex] = diarizedIndex;
                accurateIndex++;
                diarizedIndex++;
                continue;
            }

            PairGap(
                accurateTokens,
                accurateIndex,
                anchor.Value.AccurateOffset,
                diarizedTokens,
                diarizedIndex,
                anchor.Value.DiarizedOffset,
                accurateToDiarized,
                ref exactMatches);
            accurateIndex += anchor.Value.AccurateOffset;
            diarizedIndex += anchor.Value.DiarizedOffset;
        }

        PairGap(
            accurateTokens,
            accurateIndex,
            accurateTokens.Count - accurateIndex,
            diarizedTokens,
            diarizedIndex,
            diarizedTokens.Count - diarizedIndex,
            accurateToDiarized,
            ref exactMatches);

        var shortestTranscript = Math.Min(accurateTokens.Count, diarizedTokens.Count);
        var exactMatchRatio = shortestTranscript == 0
            ? 0
            : (double)exactMatches / shortestTranscript;
        var lengthRatio = (double)accurateTokens.Count / diarizedTokens.Count;
        var enoughSharedWords = shortestTranscript <= 4
            ? exactMatches >= 1
            : exactMatchRatio >= 0.35;
        var comparableLengths = lengthRatio is >= 0.5 and <= 2.0;
        if (!enoughSharedWords || !comparableLengths)
        {
            Array.Fill(accurateToDiarized, -1);
            FillUnmappedTokens(accurateToDiarized, diarizedTokens.Count);
            return RebuildWithGptTranscribeWords(
                accurateTokens,
                diarizedSegments,
                diarizedTokens,
                accurateToDiarized,
                exactMatchRatio,
                usedProportionalFallback: true,
                "The transcripts differed too much for word matching, so GPT Transcribe wording was distributed proportionally across the speaker turns.");
        }

        FillUnmappedTokens(accurateToDiarized, diarizedTokens.Count);
        return RebuildWithGptTranscribeWords(
            accurateTokens,
            diarizedSegments,
            diarizedTokens,
            accurateToDiarized,
            exactMatchRatio,
            usedProportionalFallback: false,
            "GPT Transcribe wording was aligned to GPT-4o speaker segments.");
    }

    private static DiarizedAlignmentResult RebuildWithGptTranscribeWords(
        IReadOnlyList<WordToken> accurateTokens,
        IReadOnlyList<DiarizedTextSegment> diarizedSegments,
        IReadOnlyList<SpeakerToken> diarizedTokens,
        int[] accurateToDiarized,
        double exactMatchRatio,
        bool usedProportionalFallback,
        string reason)
    {
        var wordsBySegment = Enumerable
            .Range(0, diarizedSegments.Count)
            .Select(_ => new List<string>())
            .ToArray();
        for (var index = 0; index < accurateTokens.Count; index++)
        {
            var mappedIndex = Math.Clamp(accurateToDiarized[index], 0, diarizedTokens.Count - 1);
            wordsBySegment[diarizedTokens[mappedIndex].SegmentIndex].Add(accurateTokens[index].Original);
        }

        var rebuilt = new List<DiarizedTextSegment>(diarizedSegments.Count);
        var emptySegments = 0;
        for (var index = 0; index < diarizedSegments.Count; index++)
        {
            var text = wordsBySegment[index].Count > 0
                ? string.Join(" ", wordsBySegment[index])
                : "";
            if (wordsBySegment[index].Count == 0)
            {
                emptySegments++;
            }

            rebuilt.Add(diarizedSegments[index] with { Text = text.Trim() });
        }

        return new DiarizedAlignmentResult(
            rebuilt,
            UsedAccurateText: true,
            UsedProportionalFallback: usedProportionalFallback,
            exactMatchRatio,
            emptySegments,
            reason);
    }

    private static void PairGap(
        IReadOnlyList<WordToken> accurateTokens,
        int accurateStart,
        int accurateLength,
        IReadOnlyList<SpeakerToken> diarizedTokens,
        int diarizedStart,
        int diarizedLength,
        int[] accurateToDiarized,
        ref int exactMatches)
    {
        var paired = Math.Min(accurateLength, diarizedLength);
        for (var offset = 0; offset < paired; offset++)
        {
            var accurateIndex = accurateStart + offset;
            var diarizedIndex = diarizedStart + offset;
            accurateToDiarized[accurateIndex] = diarizedIndex;
            if (SameWord(accurateTokens[accurateIndex], diarizedTokens[diarizedIndex].Token))
            {
                exactMatches++;
            }
        }
    }

    private static Anchor? FindNextAnchor(
        IReadOnlyList<WordToken> accurateTokens,
        int accurateStart,
        IReadOnlyList<SpeakerToken> diarizedTokens,
        int diarizedStart)
    {
        Anchor? best = null;
        var accurateLimit = Math.Min(AnchorLookahead, accurateTokens.Count - accurateStart - 1);
        var diarizedLimit = Math.Min(AnchorLookahead, diarizedTokens.Count - diarizedStart - 1);
        for (var accurateOffset = 0; accurateOffset <= accurateLimit; accurateOffset++)
        {
            for (var diarizedOffset = 0; diarizedOffset <= diarizedLimit; diarizedOffset++)
            {
                if (accurateOffset == 0 && diarizedOffset == 0)
                {
                    continue;
                }

                if (!SameWord(
                        accurateTokens[accurateStart + accurateOffset],
                        diarizedTokens[diarizedStart + diarizedOffset].Token))
                {
                    continue;
                }

                var run = ConsecutiveMatches(
                    accurateTokens,
                    accurateStart + accurateOffset,
                    diarizedTokens,
                    diarizedStart + diarizedOffset);
                var distance = accurateOffset + diarizedOffset;
                var score = distance - (Math.Min(run, 4) * 2) + (run == 1 ? 4 : 0);
                var candidate = new Anchor(accurateOffset, diarizedOffset, run, score);
                if (best is null || candidate.Score < best.Value.Score ||
                    (candidate.Score == best.Value.Score && candidate.Run > best.Value.Run) ||
                    (candidate.Score == best.Value.Score && candidate.Run == best.Value.Run &&
                     distance < best.Value.AccurateOffset + best.Value.DiarizedOffset))
                {
                    best = candidate;
                }
            }
        }

        return best;
    }

    private static int ConsecutiveMatches(
        IReadOnlyList<WordToken> accurateTokens,
        int accurateStart,
        IReadOnlyList<SpeakerToken> diarizedTokens,
        int diarizedStart)
    {
        var run = 0;
        while (run < 8 &&
               accurateStart + run < accurateTokens.Count &&
               diarizedStart + run < diarizedTokens.Count &&
               SameWord(accurateTokens[accurateStart + run], diarizedTokens[diarizedStart + run].Token))
        {
            run++;
        }

        return run;
    }

    private static void FillUnmappedTokens(int[] mappings, int diarizedTokenCount)
    {
        var index = 0;
        while (index < mappings.Length)
        {
            if (mappings[index] >= 0)
            {
                index++;
                continue;
            }

            var start = index;
            while (index < mappings.Length && mappings[index] < 0)
            {
                index++;
            }

            var length = index - start;
            var previous = start > 0 ? mappings[start - 1] : -1;
            var next = index < mappings.Length ? mappings[index] : -1;
            for (var offset = 0; offset < length; offset++)
            {
                int mapped;
                if (previous >= 0 && next >= 0)
                {
                    var position = (double)(offset + 1) / (length + 1);
                    mapped = (int)Math.Round(previous + ((next - previous) * position));
                }
                else if (previous >= 0)
                {
                    mapped = previous;
                }
                else if (next >= 0)
                {
                    mapped = next;
                }
                else
                {
                    mapped = (int)((long)(start + offset) * diarizedTokenCount / mappings.Length);
                }

                mappings[start + offset] = Math.Clamp(mapped, 0, diarizedTokenCount - 1);
            }
        }
    }

    private static IReadOnlyList<WordToken> Tokenize(string text) => WhitespaceToken()
        .Matches(text)
        .Select(match => new WordToken(match.Value, Normalize(match.Value)))
        .ToList();

    private static bool SameWord(WordToken first, WordToken second) =>
        first.Normalized.Length > 0 &&
        first.Normalized.Equals(second.Normalized, StringComparison.Ordinal);

    private static string Normalize(string token)
    {
        var decomposed = token.Normalize(NormalizationForm.FormD);
        var normalized = new StringBuilder(decomposed.Length);
        foreach (var character in decomposed)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(character) == UnicodeCategory.NonSpacingMark)
            {
                continue;
            }

            if (char.IsLetterOrDigit(character))
            {
                normalized.Append(char.ToLowerInvariant(character));
            }
        }

        return normalized.ToString();
    }

    [GeneratedRegex(@"\S+")]
    private static partial Regex WhitespaceToken();

    private sealed record WordToken(string Original, string Normalized);
    private sealed record SpeakerToken(WordToken Token, int SegmentIndex);
    private readonly record struct Anchor(int AccurateOffset, int DiarizedOffset, int Run, int Score);
}
