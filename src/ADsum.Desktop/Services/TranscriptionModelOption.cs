namespace ADsum.Desktop.Services;

public sealed record TranscriptionModelOption(
    string Id,
    string DisplayName,
    string Description,
    bool IncludesSpeakerDiarization,
    bool RequiresOpenAiKey,
    bool SupportsGeneralTerms)
{
    public string CapabilitySummary => IncludesSpeakerDiarization
        ? "Transcription and speaker labels"
        : "Transcription only - no speaker labels";

    public override string ToString() => DisplayName;
}

public static class TranscriptionModelCatalog
{
    public const string LocalWhisperPyannoteId = "local-whisper-pyannote";
    public const string Gpt4oTranscribeDiarizeId = "gpt-4o-transcribe-diarize";
    public const string GptTranscribeId = "gpt-transcribe";
    public const string LegacyId = "legacy";

    public static IReadOnlyList<TranscriptionModelOption> All { get; } =
    [
        new(
            LocalWhisperPyannoteId,
            "Local Whisper + Pyannote",
            "Current private local pipeline. Audio stays on this computer and speaker labels are included.",
            IncludesSpeakerDiarization: true,
            RequiresOpenAiKey: false,
            SupportsGeneralTerms: true),
        new(
            Gpt4oTranscribeDiarizeId,
            "OpenAI GPT-4o Transcribe Diarize",
            "Cloud transcription with built-in speaker labels. OpenAI does not accept vocabulary hints for this specialized model.",
            IncludesSpeakerDiarization: true,
            RequiresOpenAiKey: true,
            SupportsGeneralTerms: false),
        new(
            GptTranscribeId,
            "OpenAI GPT Transcribe",
            "New high-accuracy cloud file transcription with general-term hints. This model has no built-in speaker diarization.",
            IncludesSpeakerDiarization: false,
            RequiresOpenAiKey: true,
            SupportsGeneralTerms: true)
    ];

    public static TranscriptionModelOption Default => All[0];

    public static TranscriptionModelOption Resolve(string? id) =>
        All.FirstOrDefault(model => model.Id.Equals(id, StringComparison.OrdinalIgnoreCase)) ?? Default;

    public static string DisplayNameFor(string? id) => id?.ToLowerInvariant() switch
    {
        LocalWhisperPyannoteId => Resolve(LocalWhisperPyannoteId).DisplayName,
        Gpt4oTranscribeDiarizeId => Resolve(Gpt4oTranscribeDiarizeId).DisplayName,
        GptTranscribeId => Resolve(GptTranscribeId).DisplayName,
        LegacyId => "Legacy / previous ADsum",
        _ => "Unknown transcription model"
    };
}
