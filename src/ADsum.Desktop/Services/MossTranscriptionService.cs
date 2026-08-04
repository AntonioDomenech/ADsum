using System.Collections.Concurrent;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ADsum.Desktop.Services;

/// <summary>
/// Runs ADsum's local speech pipeline in a short-lived Python process and converts its structured
/// speaker segments into ADsum's existing transcript format. The legacy type name is retained so
/// the desktop and command-line entry points do not need a new public integration surface.
/// </summary>
public sealed class MossTranscriptionService : IDisposable
{
    private const int DefaultBatchSize = 8;
    private const int ErrorAccessDenied = 5;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private readonly RecordingMossResourceCoordinator _resourceCoordinator;
    private readonly bool _allowExternalJobFallback;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly ConcurrentDictionary<int, ActiveWorker> _activeWorkers = new();
    private int _disposed;

    public MossTranscriptionService(
        RecordingMossResourceCoordinator? resourceCoordinator = null,
        bool allowExternalJobFallback = false)
    {
        _resourceCoordinator = resourceCoordinator ?? RecordingMossResourceCoordinator.Shared;
        _allowExternalJobFallback = allowExternalJobFallback;
    }

    public async Task<string> TranscribeAsync(
        string audioPath,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default,
        string? diagnosticsPath = null)
    {
        ThrowIfDisposed();
        if (!File.Exists(audioPath))
        {
            throw new FileNotFoundException("Audio file was not found.", audioPath);
        }

        var fullAudioPath = Path.GetFullPath(audioPath);
        var requestId = Guid.NewGuid().ToString("N");
        var jobDirectory = Path.Combine(Path.GetTempPath(), "ADsum", "LocalSpeech", requestId);
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
                    progress?.Report("Local transcription is waiting for recording to finish");
                }

                var lease = await _resourceCoordinator
                    .AcquireMossLeaseAsync(linkedCancellation.Token)
                    .ConfigureAwait(false);
                var wasPreempted = false;
                string? resultPath = null;
                try
                {
                    progress?.Report("Starting local transcription");
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
                    progress?.Report("Local transcription paused so recording has full computer power; completed work is saved");
                    await _resourceCoordinator
                        .WaitForRecordingToEndAsync(linkedCancellation.Token)
                        .ConfigureAwait(false);
                    progress?.Report("Recording finished; restarting local transcription");
                    continue;
                }

                var workerResultPath = resultPath ?? outputPath;
                var result = await ReadResultAsync(
                        workerResultPath,
                        progress,
                        linkedCancellation.Token)
                    .ConfigureAwait(false);
                if (!string.IsNullOrWhiteSpace(diagnosticsPath))
                {
                    CopyDiagnostics(workerResultPath, diagnosticsPath);
                }
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
        LocalSpeechWorkerRequest request,
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
                throw new InvalidOperationException("The local speech worker could not be started.");
            }
        }
        catch (Exception ex) when (ex is not InvalidOperationException)
        {
            throw new InvalidOperationException(
                $"The local speech worker could not be started with Python at '{pythonPath}'.",
                ex);
        }

        var needsProcessTreeFallback = false;
        try
        {
            job.AssignProcessTree(process.Id);
        }
        catch (Win32Exception ex) when (
            _allowExternalJobFallback && ex.NativeErrorCode == ErrorAccessDenied)
        {
            // A headless recovery command can itself be launched inside a
            // restrictive outer Job Object (for example by an automation
            // host). Its exclusive ADsum instance lock prevents recording
            // beside it, so cancellation falls back to killing the process
            // tree directly without weakening the normal desktop path.
            needsProcessTreeFallback = true;
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

        var activeWorker = new ActiveWorker(process, job, needsProcessTreeFallback);
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
                    ?? $"The local speech worker stopped with exit code {process.ExitCode}.";
                throw new InvalidOperationException(error);
            }

            if (!state.Completed)
            {
                throw new InvalidOperationException(
                    "The local speech worker exited without confirming that transcription completed.");
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
                "Local transcription was interrupted for recording.",
                ex,
                cancellationToken);
        }
        catch (Exception) when (needsProcessTreeFallback)
        {
            // When an outer Job Object prevented assignment, disposing ADsum's
            // empty job cannot stop Python. Clean it up explicitly if an
            // unexpected managed or pipe failure unwinds this method.
            activeWorker.Terminate();
            await activeWorker.WaitForExitAsync().ConfigureAwait(false);
            await DrainWorkerPipesAsync(stdoutTask, stderrTask, TimeSpan.FromSeconds(5)).ConfigureAwait(false);
            throw;
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
                        progress?.Report(
                            "Loading local speech models" +
                            FormatTimingSuffix(root, state.Elapsed.Elapsed.TotalSeconds));
                        break;
                    case "progress":
                        ReportWorkerProgress(root, progress, state.Elapsed.Elapsed.TotalSeconds);
                        break;
                    case "stage_started":
                        ReportWorkerStage(root, progress, completed: false, state.Elapsed.Elapsed.TotalSeconds);
                        break;
                    case "stage_completed":
                        ReportWorkerStage(root, progress, completed: true, state.Elapsed.Elapsed.TotalSeconds);
                        break;
                    case "timing":
                        ReportWorkerTiming(root, progress, state.Elapsed.Elapsed.TotalSeconds);
                        break;
                    case "chunk_started":
                        ReportChunkProgress(root, progress, completed: false, state.Elapsed.Elapsed.TotalSeconds);
                        break;
                    case "chunk_completed":
                        ReportChunkProgress(root, progress, completed: true, state.Elapsed.Elapsed.TotalSeconds);
                        break;
                    case "completed":
                        state.Elapsed.Stop();
                        state.Completed = true;
                        state.ResultPath = ReadString(root, "resultPath")
                            ?? ReadString(root, "result_path")
                            ?? state.ResultPath;
                        var timingSuffix = FormatTimingSuffix(root, state.Elapsed.Elapsed.TotalSeconds);
                        if (timingSuffix.Length > 0)
                        {
                            progress?.Report("Local transcription completed" + timingSuffix);
                        }
                        break;
                    case "error":
                        var code = ReadString(root, "code");
                        var message = ReadString(root, "message") ?? "Local transcription failed.";
                        state.ErrorMessage = string.IsNullOrWhiteSpace(code)
                            ? message
                            : $"Local speech {code}: {message}";
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

    private static void ReportWorkerProgress(
        JsonElement root,
        IProgress<string>? progress,
        double fallbackElapsedSeconds)
    {
        var rawPhase = ReadString(root, "phase");
        var phase = FormatPhase(rawPhase);
        if (string.IsNullOrWhiteSpace(phase))
        {
            return;
        }

        if (rawPhase?.Equals("batch_size_fallback", StringComparison.OrdinalIgnoreCase) == true)
        {
            var failedBatchSize = ReadInt(root, "failedBatchSize");
            var nextBatchSize = ReadInt(root, "nextBatchSize");
            if (failedBatchSize is not null && nextBatchSize is not null)
            {
                phase += $" ({failedBatchSize.Value} to {nextBatchSize.Value})";
            }
        }

        if (TryReadDouble(root, "progress", out var amount) ||
            TryReadDouble(root, "percent", out amount))
        {
            if (amount <= 1)
            {
                amount *= 100;
            }
            progress?.Report(
                $"Local transcription: {phase} ({Math.Clamp(amount, 0, 100):F0}%)" +
                FormatTimingSuffix(root, fallbackElapsedSeconds));
            return;
        }

        progress?.Report(
            $"Local transcription: {phase}" +
            FormatTimingSuffix(root, fallbackElapsedSeconds));
    }

    private static void ReportWorkerStage(
        JsonElement root,
        IProgress<string>? progress,
        bool completed,
        double fallbackElapsedSeconds)
    {
        var phase = ReadString(root, "stage")
            ?? ReadString(root, "phase")
            ?? ReadString(root, "name");
        phase = FormatPhase(phase);
        if (string.IsNullOrWhiteSpace(phase))
        {
            return;
        }

        var action = completed ? "finished" : "starting";
        progress?.Report(
            $"Local transcription: {action} {phase}" +
            FormatTimingSuffix(root, fallbackElapsedSeconds));
    }

    private static void ReportWorkerTiming(
        JsonElement root,
        IProgress<string>? progress,
        double fallbackElapsedSeconds)
    {
        var phase = ReadString(root, "stage")
            ?? ReadString(root, "phase")
            ?? "processing";
        progress?.Report(
            $"Local transcription: {FormatPhase(phase)}" +
            FormatTimingSuffix(root, fallbackElapsedSeconds));
    }

    private static string? FormatPhase(string? phase)
    {
        if (string.IsNullOrWhiteSpace(phase))
        {
            return null;
        }

        var normalized = phase.Trim().ToLowerInvariant();
        return normalized switch
        {
            "inspecting_audio" => "checking recording",
            "loading_asr_model" => "loading transcription model",
            "transcribing_audio" => "transcribing recording",
            "batch_size_fallback" => "reducing GPU batch size",
            "asr_completed" => "transcription pass completed",
            "releasing_asr_model" => "freeing transcription memory",
            "loading_diarization_model" => "loading speaker model",
            "diarizing_audio" => "identifying speakers",
            "merging_speakers" => "joining words with speakers",
            "writing_result" => "saving transcript",
            _ => normalized.Replace('_', ' ')
        };
    }

    private static void ReportChunkProgress(
        JsonElement root,
        IProgress<string>? progress,
        bool completed,
        double fallbackElapsedSeconds)
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

        if (count == 1)
        {
            progress?.Report(
                $"Local transcription: {action} complete recording" +
                FormatTimingSuffix(root, fallbackElapsedSeconds));
            return;
        }

        if (index is not null && count is not null)
        {
            var displayIndex = zeroBasedIndex is not null ? zeroBasedIndex.Value + 1 : Math.Max(1, index.Value);
            progress?.Report(
                $"Local transcription: {action} audio part {displayIndex} of {count.Value}" +
                FormatTimingSuffix(root, fallbackElapsedSeconds));
        }
        else if (index is not null)
        {
            progress?.Report(
                $"Local transcription: {action} audio part {index.Value}" +
                FormatTimingSuffix(root, fallbackElapsedSeconds));
        }
        else
        {
            progress?.Report(
                $"Local transcription: {action} audio part" +
                FormatTimingSuffix(root, fallbackElapsedSeconds));
        }
    }

    private static async Task<LocalSpeechWorkerResult> ReadResultAsync(
        string resultPath,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(resultPath))
        {
            throw new FileNotFoundException("The local speech worker did not create its result file.", resultPath);
        }

        await using var stream = File.OpenRead(resultPath);
        var result = await JsonSerializer.DeserializeAsync<LocalSpeechWorkerResult>(
                stream,
                JsonOptions,
                cancellationToken)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("The local speech result file was empty.");

        if (result.Coverage is { Complete: false })
        {
            throw new InvalidOperationException(
                $"Local transcription stopped before the complete recording was covered " +
                $"(through {result.Coverage.CoveredUntil:F1} seconds). " +
                "The saved checkpoints can be reused when the transcription is retried.");
        }

        foreach (var warning in result.Warnings ?? new List<JsonElement>())
        {
            var warningText = FormatWarning(warning);
            if (!string.IsNullOrWhiteSpace(warningText))
            {
                progress?.Report($"Local transcription warning: {warningText}");
            }
        }

        result.Segments = (result.Segments ?? new List<LocalSpeechSegment>())
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

    private static LocalSpeechWorkerRequest BuildRequest(
        string requestId,
        string audioPath,
        string outputPath,
        string checkpointDirectory)
    {
        var batchSize = ReadIntegerSetting(
            "ADSUM_LOCAL_SPEECH_BATCH_SIZE",
            DefaultBatchSize,
            minimum: 2,
            maximum: 8);
        if (batchSize is not 8 and not 4 and not 2)
        {
            throw new InvalidOperationException(
                "ADSUM_LOCAL_SPEECH_BATCH_SIZE must be 8, 4, or 2.");
        }

        return new LocalSpeechWorkerRequest(
            ProtocolVersion: 1,
            RequestId: requestId,
            AudioPath: audioPath,
            OutputPath: outputPath,
            CheckpointDirectory: checkpointDirectory,
            Language: LocalSpeechLanguage(),
            Hotwords: LocalSpeechHotwords(),
            AsrModelPath: ResolveModelPath(
                "ADSUM_LOCAL_SPEECH_ASR_MODEL",
                Path.Combine("Models", "FasterWhisper", "large-v3-turbo")),
            DiarizationModelPath: ResolveModelPath(
                "ADSUM_LOCAL_SPEECH_DIARIZATION_MODEL",
                Path.Combine("Models", "Pyannote", "speaker-diarization-community-1")),
            MockInference: ReadBooleanSetting(
                false,
                "ADSUM_LOCAL_SPEECH_MOCK_INFERENCE",
                "ADSUM_MOSS_MOCK_INFERENCE") ? true : null,
            BatchSize: batchSize,
            Device: LocalSpeechDevice(),
            ComputeType: LocalSpeechComputeType(),
            VadFilter: ReadBooleanSetting(true, "ADSUM_LOCAL_SPEECH_VAD_FILTER"),
            WordTimestamps: true,
            RecordingComplete: true,
            Resume: true);
    }

    private static string ResolvePythonPath()
    {
        var configured = Setting("ADSUM_LOCAL_SPEECH_PYTHON", "ADSUM_MOSS_PYTHON");
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
                "The private local speech Python runtime is not installed. " +
                $"Expected Python at '{path}'. Run the ADsum local speech runtime setup first, " +
                "or set ADSUM_LOCAL_SPEECH_PYTHON to its python.exe.",
                path);
        }
        return path;
    }

    private static string ResolveWorkerPath()
    {
        var configured = Setting("ADSUM_LOCAL_SPEECH_WORKER");
        var candidates = new[]
        {
            configured,
            Path.Combine(AppContext.BaseDirectory, "Moss", "local_speech_worker.py"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Moss", "local_speech_worker.py")),
            Path.Combine(Directory.GetCurrentDirectory(), "src", "ADsum.Desktop", "Moss", "local_speech_worker.py")
        };
        var path = candidates
            .Where(candidate => !string.IsNullOrWhiteSpace(candidate))
            .Select(candidate => Path.GetFullPath(Environment.ExpandEnvironmentVariables(candidate!)))
            .FirstOrDefault(File.Exists);
        if (path is null)
        {
            throw new FileNotFoundException(
                "The local speech worker was not found. Expected Moss\\local_speech_worker.py beside ADsum.exe, " +
                "or set ADSUM_LOCAL_SPEECH_WORKER to its location.");
        }
        return path;
    }

    private static string? ResolveModelPath(string settingName, string relativeDefaultPath)
    {
        var configured = Setting(settingName);
        if (configured is not null)
        {
            var configuredPath = Path.GetFullPath(Environment.ExpandEnvironmentVariables(configured));
            if (!Directory.Exists(configuredPath))
            {
                throw new DirectoryNotFoundException(
                    $"{settingName} points to a missing directory: '{configuredPath}'.");
            }
            return configuredPath;
        }

        var localSnapshot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ADsum",
            "MossRuntime",
            relativeDefaultPath);
        return Directory.Exists(localSnapshot) ? localSnapshot : null;
    }

    private static string LocalSpeechLanguage()
    {
        var language = Setting("ADSUM_LOCAL_SPEECH_LANGUAGE", "ADSUM_MOSS_LANGUAGE")?.ToLowerInvariant() ?? "auto";
        return language is "auto" or "en" or "es" or "mixed"
            ? language
            : throw new InvalidOperationException(
                "ADSUM_LOCAL_SPEECH_LANGUAGE must be auto, en, es, or mixed.");
    }

    private static string LocalSpeechDevice()
    {
        var device = Setting("ADSUM_LOCAL_SPEECH_DEVICE")?.ToLowerInvariant() ?? "cuda";
        return device is "auto" or "cuda" or "cpu"
            ? device
            : throw new InvalidOperationException(
                "ADSUM_LOCAL_SPEECH_DEVICE must be auto, cuda, or cpu.");
    }

    private static string LocalSpeechComputeType()
    {
        var computeType = Setting("ADSUM_LOCAL_SPEECH_COMPUTE_TYPE")?.ToLowerInvariant() ?? "int8_float16";
        return computeType is "float16" or "float32" or "int8" or "int8_float16" or "int8_float32"
            ? computeType
            : throw new InvalidOperationException(
                "ADSUM_LOCAL_SPEECH_COMPUTE_TYPE must be float16, float32, int8, int8_float16, or int8_float32.");
    }

    private static IReadOnlyList<string> LocalSpeechHotwords()
    {
        var value = Setting("ADSUM_LOCAL_SPEECH_HOTWORDS", "ADSUM_MOSS_HOTWORDS");
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
                    "ADSUM_LOCAL_SPEECH_HOTWORDS contains invalid JSON.",
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
            "LocalSpeech",
            hash);
    }

    private static string FormatDiarizedTranscript(IReadOnlyList<LocalSpeechSegment> segments)
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

    private static string FormatTimingSuffix(JsonElement element, double? fallbackElapsedSeconds = null)
    {
        var details = new List<string>();
        var elapsed = ReadFirstDouble(
            element,
            "elapsedSeconds",
            "elapsed_seconds",
            "totalElapsedSeconds",
            "total_elapsed_seconds");
        elapsed ??= fallbackElapsedSeconds;
        if (elapsed is >= 0 && double.IsFinite(elapsed.Value))
        {
            details.Add($"elapsed {FormatDuration(elapsed.Value)}");
        }

        var stageElapsed = ReadFirstDouble(element, "stageElapsedSeconds", "stage_elapsed_seconds");
        if (stageElapsed is >= 0 && double.IsFinite(stageElapsed.Value))
        {
            details.Add($"this step {FormatDuration(stageElapsed.Value)}");
        }

        var eta = ReadFirstDouble(
            element,
            "etaSeconds",
            "eta_seconds",
            "remainingSeconds",
            "remaining_seconds");
        if (eta is >= 0 && double.IsFinite(eta.Value))
        {
            details.Add($"ETA {FormatDuration(eta.Value)}");
        }

        var realTimeFactor = ReadFirstDouble(element, "realTimeFactor", "real_time_factor", "rtf");
        if (realTimeFactor is >= 0 && double.IsFinite(realTimeFactor.Value))
        {
            details.Add($"{realTimeFactor.Value:F2}x real time");
        }

        return details.Count == 0 ? "" : " · " + string.Join(" · ", details);
    }

    private static double? ReadFirstDouble(JsonElement element, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (TryReadDouble(element, propertyName, out var value))
            {
                return value;
            }
        }
        return null;
    }

    private static string FormatDuration(double seconds)
    {
        var value = TimeSpan.FromSeconds(Math.Min(seconds, TimeSpan.MaxValue.TotalSeconds));
        return value.TotalHours >= 1
            ? value.ToString(@"h\:mm\:ss", CultureInfo.InvariantCulture)
            : value.ToString(@"m\:ss", CultureInfo.InvariantCulture);
    }

    private static int? ReadInt(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.TryGetInt32(out var number)
            ? number
            : null;
    }

    private static bool TryReadBoolean(JsonElement element, string propertyName, out bool value)
    {
        value = false;
        if (!element.TryGetProperty(propertyName, out var property) ||
            property.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            return false;
        }
        value = property.GetBoolean();
        return true;
    }

    private static string? Setting(params string[] names)
    {
        foreach (var name in names)
        {
            var value = NonEmpty(Environment.GetEnvironmentVariable(name));
            if (value is not null)
            {
                return value;
            }
        }
        return null;
    }

    private static bool ReadBooleanSetting(bool fallback, params string[] names)
    {
        var value = Setting(names);
        if (value is null)
        {
            return fallback;
        }
        if (value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("yes", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
        if (value.Equals("0", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("false", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("no", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        throw new InvalidOperationException($"{names[0]} must be true or false.");
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

    private sealed class ActiveWorker(
        Process process,
        WindowsKillOnCloseJob job,
        bool needsProcessTreeFallback)
    {
        public Process Process { get; } = process;

        public WindowsKillOnCloseJob Job { get; } = job;

        public bool NeedsProcessTreeFallback { get; } = needsProcessTreeFallback;

        public void Terminate()
        {
            if (NeedsProcessTreeFallback)
            {
                try
                {
                    Job.Terminate();
                }
                catch
                {
                    // The outer Job Object may prevent this worker from being
                    // a member of ADsum's job; the process-tree kill below is
                    // the required fallback in that case.
                }
                TryKillProcessTree(Process);
                return;
            }

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

    private static void CopyDiagnostics(string sourcePath, string destinationPath)
    {
        var fullDestinationPath = Path.GetFullPath(destinationPath);
        var directory = Path.GetDirectoryName(fullDestinationPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }
        File.Copy(sourcePath, fullDestinationPath, overwrite: true);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    private sealed record LocalSpeechWorkerRequest(
        int ProtocolVersion,
        string RequestId,
        string AudioPath,
        string OutputPath,
        string CheckpointDirectory,
        string Language,
        IReadOnlyList<string> Hotwords,
        string? AsrModelPath,
        string? DiarizationModelPath,
        bool? MockInference,
        int BatchSize,
        string Device,
        string ComputeType,
        bool VadFilter,
        bool WordTimestamps,
        bool RecordingComplete,
        bool Resume);

    private sealed class WorkerEventState(string defaultResultPath)
    {
        public bool Completed { get; set; }

        public string? ErrorMessage { get; set; }

        public string? ResultPath { get; set; } = defaultResultPath;

        public Stopwatch Elapsed { get; } = Stopwatch.StartNew();

        public List<string> UnstructuredOutput { get; } = new();
    }

    private sealed class LocalSpeechWorkerResult
    {
        public List<LocalSpeechSegment> Segments { get; set; } = new();

        public LocalSpeechCoverage? Coverage { get; set; }

        public List<JsonElement> Warnings { get; set; } = new();
    }

    private sealed class LocalSpeechCoverage
    {
        public bool Complete { get; set; }

        public double CoveredUntil { get; set; }
    }

    private sealed class LocalSpeechSegment
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
