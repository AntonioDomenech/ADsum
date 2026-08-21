namespace ADsum.Desktop.Services;

public sealed class TranscriptionRouter : IDisposable
{
    private readonly AudioCompressionService _compression;
    private readonly MossTranscriptionService _local;
    private readonly OpenAiTranscriptionService _openAi;

    public TranscriptionRouter(
        RecordingMossResourceCoordinator? resourceCoordinator = null,
        bool allowExternalJobFallback = false,
        AudioCompressionService? compression = null)
    {
        _compression = compression ?? new AudioCompressionService();
        _local = new MossTranscriptionService(resourceCoordinator, allowExternalJobFallback);
        _openAi = new OpenAiTranscriptionService();
    }

    public async Task<TranscriptionRunResult> TranscribeAsync(
        string originalAudioPath,
        TranscriptionModelOption model,
        IReadOnlyList<string> generalTerms,
        string? apiKey,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default,
        string? diagnosticsPath = null)
    {
        var compressedPath = await _compression
            .EnsureCompressedAsync(originalAudioPath, progress, cancellationToken)
            .ConfigureAwait(false);

        string text;
        switch (model.Id)
        {
            case TranscriptionModelCatalog.LocalWhisperPyannoteId:
                text = await _local.TranscribeAsync(
                        compressedPath,
                        progress,
                        cancellationToken,
                        diagnosticsPath,
                        generalTerms)
                    .ConfigureAwait(false);
                break;
            case TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId:
            case TranscriptionModelCatalog.GptTranscribeId:
                text = await _openAi.TranscribeAsync(
                        compressedPath,
                        apiKey,
                        model.Id,
                        generalTerms,
                        progress,
                        cancellationToken)
                    .ConfigureAwait(false);
                break;
            default:
                throw new InvalidOperationException($"Unsupported transcription model: {model.Id}");
        }

        return new TranscriptionRunResult(text, compressedPath, model);
    }

    public void Dispose() => _local.Dispose();
}

public sealed record TranscriptionRunResult(
    string Text,
    string CompressedAudioPath,
    TranscriptionModelOption Model);
