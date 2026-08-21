using System.IO;
using System.Windows;
using System.Text.Json;
using ADsum.Desktop.Services;

namespace ADsum.Desktop;

public partial class App : Application
{
    private SingleInstanceMarker? _instanceMarker;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        if (RequiresExclusiveInstance(e.Args))
        {
            _instanceMarker = SingleInstanceMarker.TryCreate();
            if (_instanceMarker is null)
            {
                await ReportExclusiveInstanceConflictAsync(e.Args);
                return;
            }
        }

        if (HasArgument(e.Args, "--list-devices"))
        {
            await WriteDeviceListAsync(e.Args);
            return;
        }

        if (HasArgument(e.Args, "--smoke-test"))
        {
            await RunSmokeTestAsync(e.Args);
            return;
        }

        if (HasArgument(e.Args, "--compress-recordings"))
        {
            await CompressRecordingsAsync(e.Args);
            return;
        }

        if (HasArgument(e.Args, "--transcribe-meeting"))
        {
            await TranscribeMeetingAsync(e.Args);
            return;
        }

        if (HasArgument(e.Args, "--transcribe-file"))
        {
            await TranscribeFileAsync(e.Args);
            return;
        }

        if (HasArgument(e.Args, "--minutes-file"))
        {
            await CreateMinutesFileAsync(e.Args);
            return;
        }

        new MainWindow().Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _instanceMarker?.Dispose();
        _instanceMarker = null;
        base.OnExit(e);
    }

    private static bool RequiresExclusiveInstance(string[] args)
    {
        if (HasArgument(args, "--transcribe-meeting") ||
            HasArgument(args, "--transcribe-file") ||
            HasArgument(args, "--compress-recordings") ||
            HasArgument(args, "--smoke-test"))
        {
            return true;
        }

        return !HasArgument(args, "--list-devices") && !HasArgument(args, "--minutes-file");
    }

    private static async Task ReportExclusiveInstanceConflictAsync(string[] args)
    {
        const string error =
            "Another ADsum v3.2 process is already using recording or transcription. " +
            "Use that window, or wait for its offline transcription to finish.";

        if (HasArgument(args, "--transcribe-meeting") ||
            HasArgument(args, "--transcribe-file") ||
            HasArgument(args, "--compress-recordings") ||
            HasArgument(args, "--smoke-test"))
        {
            var defaultName = HasArgument(args, "--transcribe-meeting")
                ? "adsum-meeting-transcription-result.json"
                : HasArgument(args, "--transcribe-file")
                    ? "adsum-transcription-result.json"
                    : HasArgument(args, "--compress-recordings")
                        ? "adsum-compression-result.json"
                        : "adsum-smoke-result.json";
            var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), defaultName);
            var meetingDirectory = ArgValue(args, "--transcribe-meeting");
            if (!string.IsNullOrWhiteSpace(meetingDirectory) &&
                Directory.Exists(meetingDirectory) &&
                IsWithinDirectory(resultPath, meetingDirectory))
            {
                resultPath = Path.Combine(Path.GetTempPath(), defaultName);
            }
            try
            {
                EnsureParentDirectory(resultPath);
                await File.WriteAllTextAsync(
                    resultPath,
                    JsonSerializer.Serialize(
                        new { ok = false, error },
                        new JsonSerializerOptions { WriteIndented = true }));
            }
            finally
            {
                Current.Shutdown(2);
            }
            return;
        }

        MessageBox.Show(
            error,
            "ADsum v3.2 is already running",
            MessageBoxButton.OK,
            MessageBoxImage.Information);
        Current.Shutdown(2);
    }

    private static async Task CompressRecordingsAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-compression-result.json");
        try
        {
            var library = new MeetingLibraryService();
            var compression = new AudioCompressionService();
            var result = await compression.CompressLibraryAsync(library.RootDirectory);
            var payload = new
            {
                ok = result.Failed == 0,
                root = library.RootDirectory,
                result.Total,
                result.Converted,
                result.Reused,
                result.Failed,
                failures = result.Failures
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(
                resultPath,
                JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(result.Failed == 0 ? 0 : 1);
        }
        catch (Exception ex)
        {
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(
                resultPath,
                JsonSerializer.Serialize(new { ok = false, error = ex.ToString() }, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static async Task TranscribeMeetingAsync(string[] args)
    {
        var defaultResultPath = Path.Combine(Path.GetTempPath(), "adsum-meeting-transcription-result.json");
        var resultPath = ArgValue(args, "--result") ?? defaultResultPath;
        try
        {
            var requestedDirectory = ArgValue(args, "--transcribe-meeting")
                ?? throw new InvalidOperationException("Pass an ADsum meeting folder after --transcribe-meeting.");
            var meetingDirectory = Path.GetFullPath(requestedDirectory);
            if (!Directory.Exists(meetingDirectory))
            {
                throw new DirectoryNotFoundException($"The ADsum meeting folder was not found: '{meetingDirectory}'.");
            }

            if (IsWithinDirectory(resultPath, meetingDirectory))
            {
                resultPath = defaultResultPath;
                throw new InvalidOperationException(
                    "--result must be outside the meeting folder because ADsum may rename that folder after transcription.");
            }
            var diagnosticsPath = ArgValue(args, "--diagnostics");
            if (!string.IsNullOrWhiteSpace(diagnosticsPath))
            {
                diagnosticsPath = Path.GetFullPath(diagnosticsPath);
                EnsureOutputOutsideMeeting(diagnosticsPath, meetingDirectory, "--diagnostics");
                if (SamePath(resultPath, diagnosticsPath))
                {
                    throw new InvalidOperationException(
                        "--result and --diagnostics must point to different files.");
                }
            }

            var item = new MeetingLibraryService()
                .GetMeetings()
                .FirstOrDefault(candidate => SamePath(candidate.DirectoryPath, meetingDirectory))
                ?? throw new InvalidOperationException(
                    "The folder is not an ADsum meeting in the local Library.");
            if (item.RecordingPath is null || !File.Exists(item.RecordingPath))
            {
                throw new FileNotFoundException("The selected ADsum meeting has no saved recording.", item.RecordingPath);
            }

            var mixedMetrics = MeetingRecorder.MeasureWaveFile(item.RecordingPath);
            var source = new RecordingResult(
                item.Topic,
                item.DirectoryPath,
                item.StartedAt ?? item.LastWriteTime,
                mixedMetrics.Duration,
                null,
                null,
                item.RecordingPath,
                item.TranscriptPath,
                item.MinutesPath,
                new TrackMetrics(null, TimeSpan.Zero, 0, 0),
                new TrackMetrics(null, TimeSpan.Zero, 0, 0),
                mixedMetrics);

            var settings = new SettingsStore();
            var model = TranscriptionModelCatalog.Resolve(ArgValue(args, "--model") ?? settings.SelectedTranscriptionModel.Id);
            using var transcription = new TranscriptionRouter(
                allowExternalJobFallback: true);
            var run = await transcription.TranscribeAsync(
                item.RecordingPath,
                model,
                settings.GeneralTerms,
                settings.OpenAiKey,
                diagnosticsPath: diagnosticsPath);
            var transcript = run.Text;

            string? generatedTopic = null;
            var namedLocally = false;
            if (MeetingArtifactStore.NeedsGeneratedTopic(source.Name))
            {
                if (settings.UseLocalTopicNaming)
                {
                    generatedTopic = MeetingTopicFallback.FromTranscript(transcript, source.StartedAt);
                    namedLocally = true;
                }
                else
                {
                    try
                    {
                        generatedTopic = await new OpenAiMeetingMinutesService().CreateTopicAsync(
                            transcript,
                            settings.OpenAiKey,
                            settings.NotesModel);
                    }
                    catch
                    {
                        generatedTopic = MeetingTopicFallback.FromTranscript(transcript, source.StartedAt);
                        namedLocally = true;
                    }
                }
            }

            var saved = MeetingArtifactStore.SaveTranscript(
                source,
                transcript,
                model,
                run.CompressedAudioPath,
                settings.GeneralTerms,
                generatedTopic);
            var payload = new
            {
                ok = true,
                sessionDirectory = saved.SessionDirectory,
                recordingPath = saved.MixedPath,
                transcriptPath = saved.TranscriptPath,
                compressedAudioPath = run.CompressedAudioPath,
                transcriptionModel = model.Id,
                diagnosticsPath,
                topic = saved.Name,
                topicNamedLocally = namedLocally,
                startedAt = saved.StartedAt,
                durationSeconds = saved.Duration.TotalSeconds
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(
                resultPath,
                JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(
                resultPath,
                JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static async Task CreateMinutesFileAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-minutes-result.json");
        try
        {
            var transcriptPath = ArgValue(args, "--minutes-file")
                ?? throw new InvalidOperationException("Pass a transcript path after --minutes-file.");
            var settings = new SettingsStore();
            var service = new OpenAiMeetingMinutesService();
            var minutes = await service.CreateMinutesAsync(await File.ReadAllTextAsync(transcriptPath), settings.OpenAiKey, settings.NotesModel);
            var payload = new { ok = true, minutes };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static async Task TranscribeFileAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-transcription-result.json");
        try
        {
            var audioPath = ArgValue(args, "--transcribe-file")
                ?? throw new InvalidOperationException("Pass an audio path after --transcribe-file.");
            var settings = new SettingsStore();
            var model = TranscriptionModelCatalog.Resolve(ArgValue(args, "--model") ?? settings.SelectedTranscriptionModel.Id);
            using var service = new TranscriptionRouter(
                allowExternalJobFallback: true);
            var run = await service.TranscribeAsync(
                audioPath,
                model,
                settings.GeneralTerms,
                settings.OpenAiKey);
            var payload = new
            {
                ok = true,
                text = run.Text,
                compressedAudioPath = run.CompressedAudioPath,
                transcriptionModel = model.Id
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static async Task RunSmokeTestAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-smoke-result.json");
        try
        {
            var duration = double.TryParse(ArgValue(args, "--duration"), out var value) ? value : 4.0;
            var toneDelay = double.TryParse(ArgValue(args, "--tone-delay"), out var delay) ? delay : 0.0;
            var micContains = ArgValue(args, "--mic-contains");
            var outputContains = ArgValue(args, "--output-contains");

            var devices = new AudioDeviceService();
            var microphone = PickDevice(devices.GetMicrophones(), micContains);
            var output = PickDevice(devices.GetRenderDevices(), outputContains);
            var recorder = new MeetingRecorder();
            var result = await recorder.RunDeviceTestAsync(
                "Smoke test",
                microphone.Id,
                output.Id,
                TimeSpan.FromSeconds(duration),
                TimeSpan.FromSeconds(Math.Max(0, toneDelay)));

            var payload = new
            {
                ok = true,
                microphone = microphone.Name,
                output = output.Name,
                result
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static async Task WriteDeviceListAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-devices.json");
        try
        {
            var devices = new AudioDeviceService();
            var payload = new
            {
                ok = true,
                microphones = devices.GetMicrophones(),
                outputs = devices.GetRenderDevices()
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static AudioDeviceInfo PickDevice(IReadOnlyList<AudioDeviceInfo> devices, string? contains)
    {
        if (!string.IsNullOrWhiteSpace(contains))
        {
            var match = devices.FirstOrDefault(device => device.Name.Contains(contains, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
            {
                return match;
            }
        }

        return devices.FirstOrDefault(device => device.IsDefault)
            ?? devices.FirstOrDefault()
            ?? throw new InvalidOperationException("No matching audio device was found.");
    }

    private static string? ArgValue(string[] args, string name)
    {
        for (var index = 0; index < args.Length - 1; index++)
        {
            if (args[index].Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }
        return null;
    }

    private static bool HasArgument(string[] args, string name) =>
        args.Any(value => value.Equals(name, StringComparison.OrdinalIgnoreCase));

    private static void EnsureParentDirectory(string path)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }

    private static void EnsureOutputOutsideMeeting(string path, string meetingDirectory, string argumentName)
    {
        if (IsWithinDirectory(path, meetingDirectory))
        {
            throw new InvalidOperationException(
                $"{argumentName} must be outside the meeting folder because ADsum may rename that folder after transcription.");
        }
    }

    private static bool IsWithinDirectory(string path, string directory)
    {
        var fullPath = Path.GetFullPath(path);
        var fullDirectory = Path.GetFullPath(directory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return SamePath(fullPath, fullDirectory) ||
            fullPath.StartsWith(fullDirectory + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
    }

    private static bool SamePath(string first, string second) =>
        string.Equals(
            Path.GetFullPath(first).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            Path.GetFullPath(second).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);
}
