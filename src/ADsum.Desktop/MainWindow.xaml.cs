using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using ADsum.Desktop.Services;

namespace ADsum.Desktop;

public partial class MainWindow : Window
{
    private readonly AudioDeviceService _devices = new();
    private readonly SettingsStore _settings = new();
    private readonly MeetingRecorder _recorder = new();
    private readonly RecordingMossResourceCoordinator _recordingResources = RecordingMossResourceCoordinator.Shared;
    private readonly MossTranscriptionService _transcription;
    private readonly OpenAiMeetingMinutesService _minutes = new();
    private readonly MeetingLibraryService _library = new();
    private readonly DispatcherTimer _timer;
    private readonly Dictionary<string, MeetingJob> _meetingJobs = new(StringComparer.OrdinalIgnoreCase);
    private RecordingResult? _lastResult;
    private bool _isRecorderBusy;

    public MainWindow()
    {
        _transcription = new MossTranscriptionService(_recordingResources);
        InitializeComponent();
        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _timer.Tick += Timer_Tick;
        Loaded += MainWindow_Loaded;
        MicrophoneCombo.SelectionChanged += DeviceCombo_SelectionChanged;
        OutputCombo.SelectionChanged += DeviceCombo_SelectionChanged;
    }

    private void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        SessionNameBox.Text = "";
        KeyStateText.Text = _settings.HasOpenAiKey ? "OpenAI key configured" : "OpenAI key not configured";
        RefreshDevices();
        RefreshLibrary();
        UpdateUiState();
    }

    private void RefreshDevices()
    {
        var microphones = _devices.GetMicrophones();
        var outputs = _devices.GetRenderDevices();

        MicrophoneCombo.ItemsSource = microphones;
        OutputCombo.ItemsSource = outputs;
        MicrophoneCombo.SelectedItem = microphones.FirstOrDefault(device => device.IsDefault) ?? microphones.FirstOrDefault();
        OutputCombo.SelectedItem = outputs.FirstOrDefault(device => device.IsDefault) ?? outputs.FirstOrDefault();
        UpdateWarnings();
    }

    private async void TestButton_Click(object sender, RoutedEventArgs e)
    {
        await RunRecorderActionAsync(async () =>
        {
            await _recordingResources.BeginRecordingAsync();
            try
            {
                _lastResult = null;
                ClearReviewNotes();
                RenderResult(null);
                ResultText.Text = "Running 6 second device test. Speak into the mic and confirm you hear the tone.";
                await _recorder.RunDeviceTestAsync(
                    SessionNameBox.Text,
                    SelectedMicrophoneId(),
                    SelectedOutputId(),
                    TimeSpan.FromSeconds(6));
                _lastResult = _recorder.LastResult;
                RenderResult(_lastResult);
            }
            finally
            {
                _recordingResources.EndRecording();
            }
        });
    }

    private async void RecordButton_Click(object sender, RoutedEventArgs e)
    {
        if (_isRecorderBusy || _recorder.IsRecording)
        {
            return;
        }

        _isRecorderBusy = true;
        UpdateUiState();
        try
        {
            await _recordingResources.BeginRecordingAsync();
            try
            {
                _recorder.Start(SessionNameBox.Text, SelectedMicrophoneId(), SelectedOutputId());
            }
            catch
            {
                _recordingResources.EndRecording();
                throw;
            }
            _lastResult = null;
            ClearReviewNotes();
            RenderResult(null);
            ResultText.Text = "Recording...";
            _timer.Start();
            UpdateUiState();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Unable to start recording", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _isRecorderBusy = false;
            UpdateUiState();
        }
    }

    private void StopButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _lastResult = _recorder.Stop();
            _timer.Stop();
            MicLevelBar.Value = 0;
            SystemLevelBar.Value = 0;
            ElapsedText.Text = "00:00";
            RenderResult(_lastResult);
            RefreshLibrary(_lastResult.SessionDirectory);
            UpdateUiState();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Unable to stop recording", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            if (!_recorder.IsRecording)
            {
                _recordingResources.EndRecording();
            }
        }
    }

    private async void TranscribeButton_Click(object sender, RoutedEventArgs e)
    {
        var source = _lastResult;
        if (source?.MixedPath is not { } audioPath || !File.Exists(audioPath))
        {
            MessageBox.Show(this, "Record audio before transcribing.", "No audio", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var apiKey = _settings.OpenAiKey;
        var notesModel = _settings.NotesModel;
        string transcript = "";
        string? generatedTopic = null;
        var usedLocalTopicFallback = false;
        await RunMeetingJobAsync(
            source,
            "Creating transcript",
            onStarted: () =>
            {
                if (!IsDisplayedLastResult(source.SessionDirectory))
                {
                    return;
                }

                TranscriptBox.Text = "Preparing local MOSS transcription...";
                MinutesBox.Text = "Transcript is being created. Notes are separate.";
            },
            action: async progress =>
            {
                transcript = await _transcription.TranscribeAsync(audioPath, progress);
                if (MeetingArtifactStore.NeedsGeneratedTopic(source.Name))
                {
                    try
                    {
                        generatedTopic = await _minutes.CreateTopicAsync(transcript, apiKey, notesModel, progress);
                    }
                    catch
                    {
                        generatedTopic = MeetingTopicFallback.FromTranscript(transcript, source.StartedAt);
                        usedLocalTopicFallback = true;
                        progress.Report("Naming meeting locally");
                    }
                }
            },
            onSuccess: () =>
            {
                var result = MeetingArtifactStore.SaveTranscript(source, transcript, generatedTopic);
                if (ReplaceLastResultIfMatching(source.SessionDirectory, result))
                {
                    TranscriptBox.Text = string.IsNullOrWhiteSpace(transcript) ? "(No text returned.)" : transcript.Trim();
                    TranscriptStateText.Text = usedLocalTopicFallback ? "Done - named locally" : "Done";
                }

                RefreshLibraryAfterJob(source.SessionDirectory, result.SessionDirectory);
            });
    }

    private async void CreateNotesButton_Click(object sender, RoutedEventArgs e)
    {
        var source = _lastResult;
        if (source?.TranscriptPath is not { } transcriptPath || !File.Exists(transcriptPath))
        {
            MessageBox.Show(this, "Create a transcript before generating notes.", "No transcript", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var apiKey = _settings.OpenAiKey;
        var notesModel = _settings.NotesModel;
        string minutes = "";
        await RunMeetingJobAsync(
            source,
            "Creating notes",
            onStarted: () =>
            {
                if (IsDisplayedLastResult(source.SessionDirectory))
                {
                    MinutesBox.Text = "Generating meeting notes...";
                }
            },
            action: async progress =>
            {
                var transcript = File.ReadAllText(transcriptPath);
                minutes = await _minutes.CreateMinutesAsync(transcript, apiKey, notesModel, progress);
            },
            onSuccess: () =>
            {
                var result = MeetingArtifactStore.SaveMinutes(source, minutes);
                if (ReplaceLastResultIfMatching(source.SessionDirectory, result))
                {
                    MinutesBox.Text = minutes.Trim();
                    TranscriptBox.Text = ReadTextPreview(result.TranscriptPath, "No transcript saved for this meeting.");
                    TranscriptStateText.Text = "Done";
                }

                RefreshLibraryAfterJob(source.SessionDirectory, result.SessionDirectory);
            });
    }

    private void SaveKeyButton_Click(object sender, RoutedEventArgs e)
    {
        var key = ApiKeyBox.Password.Trim();
        if (string.IsNullOrWhiteSpace(key))
        {
            MessageBox.Show(this, "Paste your OpenAI key before saving.", "Missing key", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        _settings.SaveOpenAiKey(key);
        ApiKeyBox.Clear();
        KeyStateText.Text = "OpenAI key configured";
    }

    private void RefreshButton_Click(object sender, RoutedEventArgs e) => RefreshDevices();

    private void LibraryRefreshButton_Click(object sender, RoutedEventArgs e) => RefreshLibrary();

    private void MainTabs_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ReferenceEquals(e.Source, MainTabs) && MainTabs.SelectedIndex == 1)
        {
            RefreshLibrary(SelectedLibraryMeeting()?.DirectoryPath);
        }
    }

    private void LibraryList_SelectionChanged(object sender, SelectionChangedEventArgs e) => RenderLibrarySelection(SelectedLibraryMeeting());

    private void OpenFolderButton_Click(object sender, RoutedEventArgs e)
    {
        if (_lastResult?.SessionDirectory is null || !Directory.Exists(_lastResult.SessionDirectory))
        {
            return;
        }

        OpenPath(_lastResult.SessionDirectory);
    }

    private void OpenRecordingButton_Click(object sender, RoutedEventArgs e)
    {
        if (_lastResult?.MixedPath is not null && File.Exists(_lastResult.MixedPath))
        {
            OpenPath(_lastResult.MixedPath);
        }
    }

    private void OpenMinutesButton_Click(object sender, RoutedEventArgs e)
    {
        if (_lastResult?.MinutesPath is not null && File.Exists(_lastResult.MinutesPath))
        {
            OpenPath(_lastResult.MinutesPath);
        }
    }

    private void OpenTranscriptButton_Click(object sender, RoutedEventArgs e)
    {
        if (_lastResult?.TranscriptPath is not null && File.Exists(_lastResult.TranscriptPath))
        {
            OpenPath(_lastResult.TranscriptPath);
        }
    }

    private void LibraryOpenFolderButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item is not null && Directory.Exists(item.DirectoryPath))
        {
            OpenPath(item.DirectoryPath);
        }
    }

    private void LibraryOpenRecordingButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item?.RecordingPath is not null && File.Exists(item.RecordingPath))
        {
            OpenPath(item.RecordingPath);
        }
    }

    private async void LibraryTranscribeButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item?.RecordingPath is null || !File.Exists(item.RecordingPath))
        {
            MessageBox.Show(this, "Select a saved meeting with a recording first.", "No recording", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var source = RecordingResultFromLibraryItem(item);
        var audioPath = source.MixedPath!;
        var apiKey = _settings.OpenAiKey;
        var notesModel = _settings.NotesModel;
        string transcript = "";
        string? generatedTopic = null;
        await RunMeetingJobAsync(
            source,
            "Creating transcript",
            onStarted: () =>
            {
                if (!IsSelectedLibraryMeeting(source.SessionDirectory))
                {
                    return;
                }

                LibraryTranscriptBox.Text = "Preparing local MOSS transcription...";
                LibraryMinutesBox.Text = "Transcript is being created. Notes are separate.";
            },
            action: async progress =>
            {
                transcript = await _transcription.TranscribeAsync(audioPath, progress);
                if (MeetingArtifactStore.NeedsGeneratedTopic(source.Name))
                {
                    try
                    {
                        generatedTopic = await _minutes.CreateTopicAsync(transcript, apiKey, notesModel, progress);
                    }
                    catch
                    {
                        generatedTopic = MeetingTopicFallback.FromTranscript(transcript, source.StartedAt);
                        progress.Report("Naming meeting locally");
                    }
                }
            },
            onSuccess: () =>
            {
                var result = MeetingArtifactStore.SaveTranscript(source, transcript, generatedTopic);
                if (IsSelectedLibraryMeeting(source.SessionDirectory))
                {
                    LibraryTranscriptBox.Text = string.IsNullOrWhiteSpace(transcript) ? "(No text returned.)" : transcript.Trim();
                }

                ReplaceLastResultIfMatching(source.SessionDirectory, result);
                RefreshLibraryAfterJob(source.SessionDirectory, result.SessionDirectory);
            });
    }

    private async void LibraryCreateNotesButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item?.TranscriptPath is null || !File.Exists(item.TranscriptPath))
        {
            MessageBox.Show(this, "Create a transcript before generating notes.", "No transcript", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var source = RecordingResultFromLibraryItem(item, requireRecording: false);
        var transcriptPath = source.TranscriptPath!;
        var apiKey = _settings.OpenAiKey;
        var notesModel = _settings.NotesModel;
        string minutes = "";
        await RunMeetingJobAsync(
            source,
            "Creating notes",
            onStarted: () =>
            {
                if (IsSelectedLibraryMeeting(source.SessionDirectory))
                {
                    LibraryMinutesBox.Text = "Generating meeting notes...";
                }
            },
            action: async progress =>
            {
                var transcript = File.ReadAllText(transcriptPath);
                minutes = await _minutes.CreateMinutesAsync(transcript, apiKey, notesModel, progress);
            },
            onSuccess: () =>
            {
                var result = MeetingArtifactStore.SaveMinutes(source, minutes);
                if (IsSelectedLibraryMeeting(source.SessionDirectory))
                {
                    LibraryMinutesBox.Text = minutes.Trim();
                    LibraryTranscriptBox.Text = ReadTextPreview(result.TranscriptPath, "No transcript saved for this meeting.");
                }

                ReplaceLastResultIfMatching(source.SessionDirectory, result);
                RefreshLibraryAfterJob(source.SessionDirectory, result.SessionDirectory);
            });
    }

    private void LibraryOpenMinutesButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item?.MinutesPath is not null && File.Exists(item.MinutesPath))
        {
            OpenPath(item.MinutesPath);
        }
    }

    private void LibraryOpenTranscriptButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item?.TranscriptPath is not null && File.Exists(item.TranscriptPath))
        {
            OpenPath(item.TranscriptPath);
        }
    }

    private void DeviceCombo_SelectionChanged(object sender, SelectionChangedEventArgs e) => UpdateWarnings();

    private void Timer_Tick(object? sender, EventArgs e)
    {
        MicLevelBar.Value = Math.Min(1, Math.Sqrt(_recorder.MicrophoneLevel) * 3);
        SystemLevelBar.Value = Math.Min(1, Math.Sqrt(_recorder.SystemLevel) * 3);
        ElapsedText.Text = _recorder.Elapsed.ToString(@"mm\:ss");
    }

    private void RenderResult(RecordingResult? result)
    {
        if (result is null)
        {
            ResultText.Text = "No recording yet.";
            UpdateLastRecordingButtons();
            return;
        }

        ResultText.Text =
            $"{result.Name}\n" +
            $"{result.Duration.TotalSeconds:F1}s\n\n" +
            $"Recording: {FormatMetrics(result.Mixed)}\n" +
            $"Transcript: {SavedState(result.TranscriptPath)}\n" +
            $"Minutes: {SavedState(result.MinutesPath)}\n\n" +
            $"Folder: {result.SessionDirectory}";
        UpdateLastRecordingButtons();
    }

    private static string FormatMetrics(TrackMetrics metrics)
    {
        if (metrics.Path is null)
        {
            return "missing";
        }
        return $"{metrics.Duration.TotalSeconds:F1}s, peak {metrics.Peak:F3}, rms {metrics.Rms:F3}";
    }

    private static string SavedState(string? path) => !string.IsNullOrWhiteSpace(path) && File.Exists(path) ? "saved" : "not created";

    private string SelectedMicrophoneId() => ((AudioDeviceInfo?)MicrophoneCombo.SelectedItem)?.Id ?? string.Empty;

    private string SelectedOutputId() => ((AudioDeviceInfo?)OutputCombo.SelectedItem)?.Id ?? string.Empty;

    private void UpdateWarnings()
    {
        MicrophoneWarningText.Text = ((AudioDeviceInfo?)MicrophoneCombo.SelectedItem)?.Warning ?? "";
        OutputWarningText.Text = ((AudioDeviceInfo?)OutputCombo.SelectedItem)?.Warning ?? "";
    }

    private async Task RunRecorderActionAsync(Func<Task> action)
    {
        if (_isRecorderBusy || _recorder.IsRecording)
        {
            return;
        }

        _isRecorderBusy = true;
        UpdateUiState();
        try
        {
            await action();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "ADsum", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _isRecorderBusy = false;
            UpdateUiState();
        }
    }

    private async Task RunMeetingJobAsync(
        RecordingResult source,
        string operation,
        Action onStarted,
        Func<IProgress<string>, Task> action,
        Action onSuccess)
    {
        var key = NormalizedPath(source.SessionDirectory);
        if (_meetingJobs.TryGetValue(key, out var existingJob))
        {
            MessageBox.Show(
                this,
                $"This meeting is already busy: {existingJob.Operation}. You can start work on a different recording while it finishes.",
                "Meeting already processing",
                MessageBoxButton.OK,
                MessageBoxImage.Information);
            return;
        }

        var job = new MeetingJob(operation, "Starting");
        _meetingJobs.Add(key, job);

        try
        {
            onStarted();
            UpdateUiState();
            var progress = new Progress<string>(message =>
            {
                job.Status = message;
                UpdateUiState();
            });
            await Task.Run(() => action(progress));
            job.Status = "Done";
            UpdateUiState();
            onSuccess();
        }
        catch (OperationCanceledException)
        {
            job.Status = "Cancelled";
            UpdateUiState();
        }
        catch (Exception ex)
        {
            job.Status = "Failed";
            UpdateUiState();
            if (IsDisplayedLastResult(source.SessionDirectory))
            {
                TranscriptStateText.Text = "Failed";
            }
            MessageBox.Show(this, ex.Message, $"ADsum - {operation}", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _meetingJobs.Remove(key);
            UpdateUiState();
        }
    }

    private void UpdateUiState()
    {
        UpdateHeaderState();

        var recorderControlsEnabled = !_isRecorderBusy && !_recorder.IsRecording;
        RefreshButton.IsEnabled = recorderControlsEnabled;
        TestButton.IsEnabled = recorderControlsEnabled;
        RecordButton.IsEnabled = recorderControlsEnabled;
        StopButton.IsEnabled = _recorder.IsRecording;
        SaveKeyButton.IsEnabled = true;
        LibraryRefreshButton.IsEnabled = true;
        LibraryList.IsEnabled = true;

        UpdateLastRecordingButtons();
        UpdateVisibleLastJobStatus();
        UpdateLibrarySelectionState(SelectedLibraryMeeting());
    }

    private void UpdateHeaderState()
    {
        var recorderState = _recorder.IsRecording
            ? "Recording"
            : _isRecorderBusy
                ? "Testing devices"
                : "Idle";
        StateText.Text = _meetingJobs.Count switch
        {
            0 => recorderState,
            1 => $"{recorderState} - 1 background job",
            _ => $"{recorderState} - {_meetingJobs.Count} background jobs"
        };
    }

    private void UpdateLastRecordingButtons()
    {
        var result = _lastResult;
        var meetingIsBusy = result is not null && IsMeetingJobActive(result.SessionDirectory);
        TranscribeButton.IsEnabled = !meetingIsBusy && result?.MixedPath is not null && File.Exists(result.MixedPath);
        CreateNotesButton.IsEnabled = !meetingIsBusy && result?.TranscriptPath is not null && File.Exists(result.TranscriptPath);
        OpenFolderButton.IsEnabled = !meetingIsBusy && result is not null && Directory.Exists(result.SessionDirectory);
        OpenRecordingButton.IsEnabled = !meetingIsBusy && result?.MixedPath is not null && File.Exists(result.MixedPath);
        OpenTranscriptButton.IsEnabled = !meetingIsBusy && result?.TranscriptPath is not null && File.Exists(result.TranscriptPath);
        OpenMinutesButton.IsEnabled = !meetingIsBusy && result?.MinutesPath is not null && File.Exists(result.MinutesPath);
    }

    private void UpdateVisibleLastJobStatus()
    {
        if (_lastResult is not null && TryGetMeetingJob(_lastResult.SessionDirectory, out var job))
        {
            TranscriptStateText.Text = $"{job.Operation}: {job.Status}";
        }
    }

    private void ClearReviewNotes()
    {
        TranscriptStateText.Text = "Ready";
        TranscriptBox.Text = "Record audio, then create a transcript.";
        MinutesBox.Text = "Create a transcript, then create notes.";
    }

    private void RefreshLibrary(string? preferredDirectory = null)
    {
        var selectedDirectory = preferredDirectory ?? SelectedLibraryMeeting()?.DirectoryPath;
        var meetings = _library.GetMeetings();
        LibraryList.ItemsSource = meetings;
        LibraryCountText.Text = meetings.Count == 1 ? "1 meeting" : $"{meetings.Count} meetings";

        var selected = !string.IsNullOrWhiteSpace(selectedDirectory)
            ? meetings.FirstOrDefault(item => SamePath(item.DirectoryPath, selectedDirectory))
            : null;
        selected ??= meetings.FirstOrDefault();
        LibraryList.SelectedItem = selected;
        RenderLibrarySelection(selected);
    }

    private void RenderLibrarySelection(MeetingLibraryItem? item)
    {
        if (item is null)
        {
            LibraryTitleText.Text = "No meetings found";
            LibraryDetailsText.Text = $"ADsum will list saved meetings from {_library.RootDirectory}.";
            LibraryMinutesBox.Text = "No meeting selected.";
            LibraryTranscriptBox.Text = "No meeting selected.";
            SetLibraryButtons(false, false, false, false, false, false);
            return;
        }

        LibraryTitleText.Text = item.Topic;
        LibraryMinutesBox.Text = ReadTextPreview(item.MinutesPath, "No meeting minutes saved for this meeting.");
        LibraryTranscriptBox.Text = ReadTextPreview(item.TranscriptPath, "No transcript saved for this meeting.");
        UpdateLibrarySelectionState(item);
    }

    private void UpdateLibrarySelectionState(MeetingLibraryItem? item)
    {
        if (item is null)
        {
            SetLibraryButtons(false, false, false, false, false, false);
            return;
        }

        var meetingIsBusy = IsMeetingJobActive(item.DirectoryPath);
        RenderLibraryDetails(item);
        SetLibraryButtons(
            !meetingIsBusy && Directory.Exists(item.DirectoryPath),
            !meetingIsBusy && item.RecordingPath is not null && File.Exists(item.RecordingPath),
            !meetingIsBusy && item.RecordingPath is not null && File.Exists(item.RecordingPath),
            !meetingIsBusy && item.TranscriptPath is not null && File.Exists(item.TranscriptPath),
            !meetingIsBusy && item.MinutesPath is not null && File.Exists(item.MinutesPath),
            !meetingIsBusy && item.TranscriptPath is not null && File.Exists(item.TranscriptPath));
    }

    private void RenderLibraryDetails(MeetingLibraryItem item)
    {
        var details =
            $"{item.DateText}\n" +
            $"{item.FileSummary}\n" +
            $"{item.DirectoryPath}";
        if (TryGetMeetingJob(item.DirectoryPath, out var job))
        {
            details += $"\n\n{job.Operation}: {job.Status}";
        }

        LibraryDetailsText.Text = details;
    }

    private MeetingLibraryItem? SelectedLibraryMeeting() => (MeetingLibraryItem?)LibraryList.SelectedItem;

    private void SetLibraryButtons(bool folder, bool recording, bool transcribe, bool createNotes, bool minutes, bool transcript)
    {
        LibraryOpenFolderButton.IsEnabled = folder;
        LibraryOpenRecordingButton.IsEnabled = recording;
        LibraryTranscribeButton.IsEnabled = transcribe;
        LibraryCreateNotesButton.IsEnabled = createNotes;
        LibraryOpenMinutesButton.IsEnabled = minutes;
        LibraryOpenTranscriptButton.IsEnabled = transcript;
    }

    private bool ReplaceLastResultIfMatching(string originalDirectory, RecordingResult result)
    {
        if (!IsDisplayedLastResult(originalDirectory))
        {
            return false;
        }

        _lastResult = result;
        RenderResult(result);
        return true;
    }

    private void RefreshLibraryAfterJob(string originalDirectory, string completedDirectory)
    {
        var selectedDirectory = SelectedLibraryMeeting()?.DirectoryPath;
        var preferredDirectory = string.IsNullOrWhiteSpace(selectedDirectory) || SamePath(selectedDirectory, originalDirectory)
            ? completedDirectory
            : selectedDirectory;
        RefreshLibrary(preferredDirectory);
    }

    private bool IsDisplayedLastResult(string directory) =>
        _lastResult is not null && SamePath(_lastResult.SessionDirectory, directory);

    private bool IsSelectedLibraryMeeting(string directory) =>
        SamePath(SelectedLibraryMeeting()?.DirectoryPath, directory);

    private bool IsMeetingJobActive(string directory) =>
        _meetingJobs.ContainsKey(NormalizedPath(directory));

    private bool TryGetMeetingJob(string directory, out MeetingJob job) =>
        _meetingJobs.TryGetValue(NormalizedPath(directory), out job!);

    private static string NormalizedPath(string path) =>
        Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

    private static string ReadTextPreview(string? path, string missingText)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return missingText;
        }

        try
        {
            return File.ReadAllText(path);
        }
        catch (Exception ex)
        {
            return $"Unable to read file: {ex.Message}";
        }
    }

    private static RecordingResult RecordingResultFromLibraryItem(MeetingLibraryItem item, bool requireRecording = true)
    {
        TrackMetrics mixedMetrics;
        string? mixedPath;
        if (item.RecordingPath is not null && File.Exists(item.RecordingPath))
        {
            mixedPath = item.RecordingPath;
            mixedMetrics = MeetingRecorder.MeasureWaveFile(item.RecordingPath);
        }
        else if (requireRecording)
        {
            throw new FileNotFoundException("Saved recording was not found.", item.RecordingPath);
        }
        else
        {
            mixedPath = null;
            mixedMetrics = new TrackMetrics(null, TimeSpan.Zero, 0, 0);
        }

        return new RecordingResult(
            item.Topic,
            item.DirectoryPath,
            item.StartedAt ?? item.LastWriteTime,
            mixedMetrics.Duration,
            null,
            null,
            mixedPath,
            item.TranscriptPath,
            item.MinutesPath,
            new TrackMetrics(null, TimeSpan.Zero, 0, 0),
            new TrackMetrics(null, TimeSpan.Zero, 0, 0),
            mixedMetrics);
    }

    private static bool SamePath(string? first, string? second)
    {
        if (string.IsNullOrWhiteSpace(first) || string.IsNullOrWhiteSpace(second))
        {
            return false;
        }

        return string.Equals(
            Path.GetFullPath(first).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            Path.GetFullPath(second).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);
    }

    private static void OpenPath(string path)
    {
        Process.Start(new ProcessStartInfo
        {
            FileName = path,
            UseShellExecute = true
        });
    }

    protected override void OnClosed(EventArgs e)
    {
        _transcription.Dispose();
        try
        {
            _recorder.Dispose();
        }
        catch
        {
            // Closing the app should still terminate its local inference worker.
        }
        finally
        {
            _recordingResources.EndRecording();
            base.OnClosed(e);
        }
    }

    private sealed class MeetingJob(string operation, string status)
    {
        public string Operation { get; } = operation;

        public string Status { get; set; } = status;
    }
}
