using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ADsum.Desktop.Services;

/// <summary>
/// Runs MOSS locally in a short-lived Python process and converts its structured
/// speaker segments into ADsum's existing transcript format.
/// </summary>
public sealed class MossTranscriptionService : IDisposable
{
    private const string ModelRevision = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8";
    private const int DefaultChunkSeconds = 300;
    private const int DefaultOverlapSeconds = 30;
    private const int DefaultEncoderBatchSize = 1;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private readonly RecordingMossResourceCoordinator _resourceCoordinator;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly ConcurrentDictionary<int, ActiveWorker> _activeWorkers = new();
    private int _disposed;

    public MossTranscriptionService(RecordingMossResourceCoordinator? resourceCoordinator = null)
    {
        _resourceCoordinator = resourceCoordinator ?? RecordingMossResourceCoordinator.Shared;
    }

    public async Task<string> TranscribeAsync(
        string audioPath,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (!File.Exists(audioPath))
        {
            throw new FileNotFoundException("Audio file was not found.", audioPath);
        }

        var fullAudioPath = Path.GetFullPath(audioPath);
        var requestId = Guid.NewGuid().ToString("N");
        var jobDirectory = Path.Combine(Path.GetTempPath(), "ADsum", "Moss", requestId);
        var outputPath = Path.Combine(jobDirectory, "result.json");
        var checkpointDirectory = CheckpointDirectoryFor(fullAudioPath);
        Directory.CreateDirectory(jobDirectory);
        Directory.CreateDirectory(checkpointDirectory);

        var request = BuildRequest(
            requestId,
            fullAudioPath,
            outputPath,
            checkpointDirectory);
        using var linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            _shutdown.Token);

        var completed = false;
        try
        {
            while (true)
            {
                linkedCancellation.Token.ThrowIfCancellationRequested();
                if (_resourceCoordinator.IsRecordingActive)
                {
                    progress?.Report("MOSS paused while recording; completed chunks are saved");
                }

                var lease = await _resourceCoordinator
                    .AcquireMossLeaseAsync(linkedCancellation.Token)
                    .ConfigureAwait(false);
                var wasPreempted = false;
                string? resultPath = null;
                try
                {
                    progress?.Report("Starting local MOSS transcription");
                    resultPath = await RunWorkerAsync(request, progress, lease.CancellationToken)
                        .ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (
                    lease.WasPreempted && !linkedCancellation.IsCancellationRequested)
                {
                    wasPreempted = true;
                }
                finally
                {
                    lease.Dispose();
                }

                if (wasPreempted)
                {
                    progress?.Report("MOSS paused while recording; completed chunks are saved");
                    await _resourceCoordinator
                        .WaitForRecordingToEndAsync(linkedCancellation.Token)
                        .ConfigureAwait(false);
                    progress?.Report("Recording finished; resuming MOSS from its checkpoint");
                    continue;
                }

                var result = await ReadResultAsync(
                        resultPath ?? outputPath,
                        progress,
                        linkedCancellation.Token)
                    .ConfigureAwait(false);
                completed = true;
                progress?.Report("Formatting transcript");
                return FormatDiarizedTranscript(result.Segments);
            }
        }
        finally
        {
            TryDeleteDirectory(jobDirectory);
            if (completed)
            {
                TryDeleteDirectory(checkpointDirectory);
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _shutdown.Cancel();
        foreach (var worker in _activeWorkers.Values)
        {
            worker.Terminate();
        }
        _shutdown.Dispose();
    }

    private async Task<string> RunWorkerAsync(
        MossWorkerRequest request,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var pythonPath = ResolvePythonPath();
        var workerPath = ResolveWorkerPath();
        var startInfo = new ProcessStartInfo
        {
            FileName = pythonPath,
            WorkingDirectory = Path.GetDirectoryName(workerPath)!,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden
        };
        startInfo.ArgumentList.Add(workerPath);
        startInfo.Environment["PYTHONUTF8"] = "1";
        startInfo.Environment["PYTHONUNBUFFERED"] = "1";

        using var job = WindowsKillOnCloseJob.Create();
        using var process = new Process { StartInfo = startInfo };
        try
        {
            if (!process.Start())
            {
                throw new InvalidOperationException("The local MOSS worker could not be started.");
            }
        }
        catch (Exception ex) when (ex is not InvalidOperationException)
        {
            throw new InvalidOperationException(
                $"The local MOSS worker could not be started with Python at '{pythonPath}'.",
                ex);
        }

        try
        {
            job.AssignProcessTree(process.Id);
        }
        catch
        {
            try
            {
                job.Terminate();
            }
            catch
            {
                TryKillProcessTree(process);
            }
            await WaitForProcessExitAsync(process).ConfigureAwait(false);
            throw;
        }

        var activeWorker = new ActiveWorker(process, job);
        _activeWorkers.TryAdd(process.Id, activeWorker);
        var state = new WorkerEventState(request.OutputPath);
        var stdoutTask = ReadWorkerEventsAsync(process.StandardOutput, state, progress);
        var stderrTask = process.StandardError.ReadToEndAsync();
        using var killRegistration = cancellationToken.Register(
            static value => ((ActiveWorker)value!).Terminate(),
            activeWorker);

        try
        {
            var requestJson = JsonSerializer.Serialize(request, JsonOptions);
            var requestBytes = Encoding.UTF8.GetBytes(requestJson + "\n");
            var inputStream = process.StandardInput.BaseStream;
            await inputStream.WriteAsync(requestBytes, cancellationToken).ConfigureAwait(false);
            await inputStream.FlushAsync(cancellationToken).ConfigureAwait(false);
            inputStream.Close();

            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            await stdoutTask.ConfigureAwait(false);
            var stderr = await stderrTask.ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();

            if (process.ExitCode != 0 || state.ErrorMessage is not null)
            {
                var error = state.ErrorMessage
                    ?? NonEmpty(stderr)
                    ?? $"The MOSS worker stopped with exit code {process.ExitCode}.";
                throw new InvalidOperationException(error);
            }

            if (!state.Completed)
            {
                throw new InvalidOperationException(
                    "The MOSS worker exited without confirming that transcription completed.");
            }

            return Path.GetFullPath(state.ResultPath ?? request.OutputPath);
        }
        catch (OperationCanceledException)
        {
            activeWorker.Terminate();
            await activeWorker.WaitForExitAsync().ConfigureAwait(false);
            await DrainWorkerPipesAsync(stdoutTask, stderrTask, TimeSpan.FromSeconds(5)).ConfigureAwait(false);
            throw;
        }
        catch (Exception ex) when (cancellationToken.IsCancellationRequested)
        {
            activeWorker.Terminate();
            await activeWorker.WaitForExitAsync().ConfigureAwait(false);
            await DrainWorkerPipesAsync(stdoutTask, stderrTask, TimeSpan.FromSeconds(5)).ConfigureAwait(false);
            throw new OperationCanceledException(
                "Local MOSS transcription was interrupted for recording.",
                ex,
                cancellationToken);
        }
        finally
        {
            _activeWorkers.TryRemove(process.Id, out _);
        }
    }

    private static async Task ReadWorkerEventsAsync(
        StreamReader reader,
        WorkerEventState state,
        IProgress<string>? progress)
    {
        while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                using var document = JsonDocument.Parse(line);
                var root = document.RootElement;
                var eventName = ReadString(root, "event") ?? ReadString(root, "type");
                switch (eventName?.ToLowerInvariant())
                {
                    case "started":
                        progress?.Report("Loading MOSS locally");
                        break;
                    case "progress":
                        ReportWorkerProgress(root, progress);
                        break;
                    case "chunk_started":
                        ReportChunkProgress(root, progress, completed: false);
                        break;
                    case "chunk_completed":
                        ReportChunkProgress(root, progress, completed: true);
                        break;
                    case "completed":
                        state.Completed = true;
                        state.ResultPath = ReadString(root, "resultPath") ?? state.ResultPath;
                        break;
                    case "error":
                        var code = ReadString(root, "code");
                        var message = ReadString(root, "message") ?? "Local MOSS transcription failed.";
                        state.ErrorMessage = string.IsNullOrWhiteSpace(code)
                            ? message
                            : $"MOSS {code}: {message}";
                        break;
                }
            }
            catch (JsonException)
            {
                state.UnstructuredOutput.Add(line.Trim());
                if (state.UnstructuredOutput.Count > 20)
                {
                    state.UnstructuredOutput.RemoveAt(0);
                }
            }
        }
    }

    private static void ReportWorkerProgress(JsonElement root, IProgress<string>? progress)
    {
        var phase = ReadString(root, "phase")?.Replace('_', ' ').Trim();
        if (string.IsNullOrWhiteSpace(phase))
        {
            return;
        }

        if (TryReadDouble(root, "progress", out var amount))
        {
            if (amount <= 1)
            {
                amount *= 100;
            }
            progress?.Report($"MOSS {phase} ({Math.Clamp(amount, 0, 100):F0}%)");
            return;
        }

        progress?.Report($"MOSS {phase}");
    }

    private static void ReportChunkProgress(
        JsonElement root,
        IProgress<string>? progress,
        bool completed)
    {
        var zeroBasedIndex = ReadInt(root, "index");
        var index = ReadInt(root, "chunkIndex")
            ?? ReadInt(root, "chunkNumber")
            ?? zeroBasedIndex;
        var count = ReadInt(root, "chunkCount")
            ?? ReadInt(root, "totalChunks")
            ?? ReadInt(root, "count")
            ?? ReadInt(root, "total");
        var action = completed ? "completed" : "transcribing";

        if (index is not null && count is not null)
        {
            var displayIndex = zeroBasedIndex is not null ? zeroBasedIndex.Value + 1 : Math.Max(1, index.Value);
            progress?.Report($"MOSS {action} chunk {displayIndex} of {count.Value}");
        }
        else if (index is not null)
        {
            progress?.Report($"MOSS {action} chunk {index.Value}");
        }
        else
        {
            progress?.Report($"MOSS {action} audio chunk");
        }
    }

    private static async Task<MossWorkerResult> ReadResultAsync(
        string resultPath,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(resultPath))
        {
            throw new FileNotFoundException("The MOSS worker did not create its result file.", resultPath);
        }

        await using var stream = File.OpenRead(resultPath);
        var result = await JsonSerializer.DeserializeAsync<MossWorkerResult>(
                stream,
                JsonOptions,
                cancellationToken)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("The MOSS result file was empty.");

        if (result.Coverage is { Complete: false })
        {
            throw new InvalidOperationException(
                $"MOSS stopped before the complete recording was covered " +
                $"(through {result.Coverage.CoveredUntil:F1} seconds). " +
                "The saved checkpoints can be reused when the transcription is retried.");
        }

        foreach (var warning in result.Warnings ?? new List<JsonElement>())
        {
            var warningText = FormatWarning(warning);
            if (!string.IsNullOrWhiteSpace(warningText))
            {
                progress?.Report($"MOSS warning: {warningText}");
            }
        }

        result.Segments = (result.Segments ?? new List<MossSegment>())
            .Where(segment =>
                double.IsFinite(segment.Start) &&
                double.IsFinite(segment.End) &&
                segment.Start >= 0 &&
                segment.End >= segment.Start &&
                !string.IsNullOrWhiteSpace(segment.Text))
            .OrderBy(segment => segment.Start)
            .ToList();
        return result;
    }

    private static MossWorkerRequest BuildRequest(
        string requestId,
        string audioPath,
        string outputPath,
        string checkpointDirectory)
    {
        var chunkSeconds = ReadIntegerSetting(
            "ADSUM_MOSS_CHUNK_SECONDS",
            DefaultChunkSeconds,
            minimum: 300,
            maximum: 1800);
        var overlapSeconds = ReadIntegerSetting(
            "ADSUM_MOSS_OVERLAP_SECONDS",
            DefaultOverlapSeconds,
            minimum: 0,
            maximum: Math.Min(600, chunkSeconds - 1));

        return new MossWorkerRequest(
            ProtocolVersion: 1,
            RequestId: requestId,
            AudioPath: audioPath,
            OutputPath: outputPath,
            CheckpointDirectory: checkpointDirectory,
            Language: MossLanguage(),
            Hotwords: MossHotwords(),
            ModelPath: ResolveModelPath(),
            MockInference: ReadBooleanSetting("ADSUM_MOSS_MOCK_INFERENCE") ? true : null,
            ChunkSeconds: chunkSeconds,
            OverlapSeconds: overlapSeconds,
            EncoderBatchSize: ReadIntegerSetting(
                "ADSUM_MOSS_ENCODER_BATCH_SIZE",
                DefaultEncoderBatchSize,
                minimum: 1,
                maximum: 16),
            Resume: true);
    }

    private static string ResolvePythonPath()
    {
        var configured = Setting("ADSUM_MOSS_PYTHON");
        var path = configured ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ADsum",
            "MossRuntime",
            ".venv",
            "Scripts",
            "python.exe");
        path = Path.GetFullPath(Environment.ExpandEnvironmentVariables(path));
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                "The private MOSS Python runtime is not installed. " +
                $"Expected Python at '{path}'. Run the ADsum MOSS runtime setup first, " +
                "or set ADSUM_MOSS_PYTHON to its python.exe.",
                path);
        }
        return path;
    }

    private static string ResolveWorkerPath()
    {
        var configured = Setting("ADSUM_MOSS_WORKER");
        var candidates = new[]
        {
            configured,
            Path.Combine(AppContext.BaseDirectory, "Moss", "moss_worker.py"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Moss", "moss_worker.py")),
            Path.Combine(Directory.GetCurrentDirectory(), "src", "ADsum.Desktop", "Moss", "moss_worker.py")
        };
        var path = candidates
            .Where(candidate => !string.IsNullOrWhiteSpace(candidate))
            .Select(candidate => Path.GetFullPath(Environment.ExpandEnvironmentVariables(candidate!)))
            .FirstOrDefault(File.Exists);
        if (path is null)
        {
            throw new FileNotFoundException(
                "The local MOSS worker was not found. Expected Moss\\moss_worker.py beside ADsum.exe, " +
                "or set ADSUM_MOSS_WORKER to its location.");
        }
        return path;
    }

    private static string? ResolveModelPath()
    {
        var configured = Setting("ADSUM_MOSS_MODEL_PATH");
        if (configured is not null)
        {
            var configuredPath = Path.GetFullPath(Environment.ExpandEnvironmentVariables(configured));
            if (!Directory.Exists(configuredPath))
            {
                throw new DirectoryNotFoundException(
                    $"ADSUM_MOSS_MODEL_PATH points to a missing directory: '{configuredPath}'.");
            }
            return configuredPath;
        }

        var localSnapshot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ADsum",
            "MossRuntime",
            "Models",
            "MOSS",
            ModelRevision);
        return Directory.Exists(localSnapshot) ? localSnapshot : null;
    }

    private static string MossLanguage()
    {
        var language = Setting("ADSUM_MOSS_LANGUAGE")?.ToLowerInvariant() ?? "auto";
        return language is "auto" or "en" or "es" or "mixed"
            ? language
            : throw new InvalidOperationException(
                "ADSUM_MOSS_LANGUAGE must be auto, en, es, or mixed.");
    }

    private static IReadOnlyList<string> MossHotwords()
    {
        var value = Setting("ADSUM_MOSS_HOTWORDS");
        if (value is null)
        {
            return Array.Empty<string>();
        }

        if (value.StartsWith("[", StringComparison.Ordinal))
        {
            try
            {
                return JsonSerializer.Deserialize<string[]>(value, JsonOptions)?
                    .Where(item => !string.IsNullOrWhiteSpace(item))
                    .Select(item => item.Trim())
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToArray()
                    ?? Array.Empty<string>();
            }
            catch (JsonException ex)
            {
                throw new InvalidOperationException(
                    "ADSUM_MOSS_HOTWORDS contains invalid JSON.",
                    ex);
            }
        }

        return value
            .Split(new[] { ',', ';', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static string CheckpointDirectoryFor(string audioPath)
    {
        var normalizedPath = Path.GetFullPath(audioPath).ToUpperInvariant();
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(normalizedPath)))[..24];
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ADsum",
            "MossRuntime",
            "Checkpoints",
            hash);
    }

    private static string FormatDiarizedTranscript(IReadOnlyList<MossSegment> segments)
    {
        if (segments.Count == 0)
        {
            return "";
        }

        var labels = new SpeakerLabeler();
        var output = new StringBuilder();
        foreach (var segment in segments.OrderBy(segment => segment.Start))
        {
            output
                .Append(FormatTimestamp(TimeSpan.FromSeconds(segment.Start)))
                .Append(" - ")
                .Append(FormatTimestamp(TimeSpan.FromSeconds(segment.End)))
                .Append("  ")
                .Append(labels.DisplayName(segment.Speaker))
                .Append(": ")
                .AppendLine(segment.Text.Trim());
        }
        return output.ToString().Trim();
    }

    private static string FormatTimestamp(TimeSpan value) =>
        value.TotalHours >= 1
            ? value.ToString(@"h\:mm\:ss")
            : value.ToString(@"m\:ss");

    private static string? ReadString(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
    }

    private static string? FormatWarning(JsonElement warning)
    {
        if (warning.ValueKind == JsonValueKind.String)
        {
            return NonEmpty(warning.GetString());
        }
        if (warning.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var code = ReadString(warning, "code");
        if (code == "speaker_continuity_uncertain")
        {
            var localSpeaker = ReadString(warning, "localSpeaker") ?? "a local speaker";
            var globalSpeaker = ReadString(warning, "globalSpeaker") ?? "a new speaker label";
            return $"speaker identity across a chunk boundary was uncertain; {localSpeaker} was kept as {globalSpeaker}";
        }

        return NonEmpty(code) ?? warning.GetRawText();
    }

    private static bool TryReadDouble(JsonElement element, string propertyName, out double value)
    {
        value = 0;
        return element.TryGetProperty(propertyName, out var property) && property.TryGetDouble(out value);
    }

    private static int? ReadInt(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.TryGetInt32(out var number)
            ? number
            : null;
    }

    private static string? Setting(string name) => NonEmpty(Environment.GetEnvironmentVariable(name));

    private static bool ReadBooleanSetting(string name)
    {
        var value = Setting(name);
        return value is not null &&
            (value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
             value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
             value.Equals("yes", StringComparison.OrdinalIgnoreCase));
    }

    private static int ReadIntegerSetting(string name, int fallback, int minimum, int maximum)
    {
        var value = Setting(name);
        if (value is null)
        {
            return fallback;
        }
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
        {
            throw new InvalidOperationException($"{name} must be a whole number.");
        }
        return Math.Clamp(parsed, minimum, maximum);
    }

    private static string? NonEmpty(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value.Trim().Trim('"');

    private static async Task WaitForProcessExitAsync(Process process)
    {
        if (!process.HasExited)
        {
            await process.WaitForExitAsync().ConfigureAwait(false);
        }
    }

    private static async Task DrainWorkerPipesAsync(
        Task stdoutTask,
        Task<string> stderrTask,
        TimeSpan timeout)
    {
        var drain = Task.WhenAll(ObservePipeAsync(stdoutTask), ObservePipeAsync(stderrTask));
        try
        {
            await drain.WaitAsync(timeout).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            // The Job Object is already empty, so a stale managed pipe reader must
            // not prevent recording from starting.
        }
    }

    private static async Task ObservePipeAsync(Task pipeTask)
    {
        try
        {
            await pipeTask.ConfigureAwait(false);
        }
        catch
        {
            // A worker pipe can close abruptly when the Job Object is terminated.
        }
    }

    private static void TryKillProcessTree(Process process)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
        catch
        {
            // Cancellation is best effort; the normal wait path will report a failure if needed.
        }
    }

    private sealed class ActiveWorker(Process process, WindowsKillOnCloseJob job)
    {
        public Process Process { get; } = process;

        public WindowsKillOnCloseJob Job { get; } = job;

        public void Terminate()
        {
            try
            {
                Job.Terminate();
            }
            catch
            {
                TryKillProcessTree(Process);
            }
        }

        public async Task WaitForExitAsync()
        {
            await WaitForProcessExitAsync(Process).ConfigureAwait(false);
            await Job.WaitForEmptyAsync().ConfigureAwait(false);
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
            // Temporary results and completed checkpoints are best-effort cleanup.
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    private sealed record MossWorkerRequest(
        int ProtocolVersion,
        string RequestId,
        string AudioPath,
        string OutputPath,
        string CheckpointDirectory,
        string Language,
        IReadOnlyList<string> Hotwords,
        string? ModelPath,
        bool? MockInference,
        int ChunkSeconds,
        int OverlapSeconds,
        int EncoderBatchSize,
        bool Resume);

    private sealed class WorkerEventState(string defaultResultPath)
    {
        public bool Completed { get; set; }

        public string? ErrorMessage { get; set; }

        public string? ResultPath { get; set; } = defaultResultPath;

        public List<string> UnstructuredOutput { get; } = new();
    }

    private sealed class MossWorkerResult
    {
        public List<MossSegment> Segments { get; set; } = new();

        public MossCoverage? Coverage { get; set; }

        public List<JsonElement> Warnings { get; set; } = new();
    }

    private sealed class MossCoverage
    {
        public bool Complete { get; set; }

        public double CoveredUntil { get; set; }
    }

    private sealed class MossSegment
    {
        public double Start { get; set; }

        public double End { get; set; }

        public string Speaker { get; set; } = "";

        public string Text { get; set; } = "";
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
