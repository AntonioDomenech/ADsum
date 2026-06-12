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
    private readonly OpenAiTranscriptionService _transcription = new();
    private readonly OpenAiMeetingMinutesService _minutes = new();
    private readonly MeetingLibraryService _library = new();
    private readonly DispatcherTimer _timer;
    private RecordingResult? _lastResult;
    private bool _isBusy;

    public MainWindow()
    {
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
        await RunBusyAsync(async () =>
        {
            ClearReviewNotes();
            ResultText.Text = "Running 6 second device test. Speak into the mic and confirm you hear the tone.";
            await _recorder.RunDeviceTestAsync(
                SessionNameBox.Text,
                SelectedMicrophoneId(),
                SelectedOutputId(),
                TimeSpan.FromSeconds(6));
            _lastResult = _recorder.LastResult;
            RenderResult(_lastResult);
        });
    }

    private void RecordButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _recorder.Start(SessionNameBox.Text, SelectedMicrophoneId(), SelectedOutputId());
            _lastResult = null;
            ClearReviewNotes();
            ResultText.Text = "Recording...";
            StateText.Text = "Recording";
            StopButton.IsEnabled = true;
            RecordButton.IsEnabled = false;
            TestButton.IsEnabled = false;
            RefreshButton.IsEnabled = false;
            TranscribeButton.IsEnabled = false;
            OpenFolderButton.IsEnabled = false;
            _timer.Start();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Unable to start recording", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void StopButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _lastResult = _recorder.Stop();
            _timer.Stop();
            StateText.Text = "Idle";
            StopButton.IsEnabled = false;
            RecordButton.IsEnabled = true;
            TestButton.IsEnabled = true;
            RefreshButton.IsEnabled = true;
            MicLevelBar.Value = 0;
            SystemLevelBar.Value = 0;
            ElapsedText.Text = "00:00";
            RenderResult(_lastResult);
            RefreshLibrary(_lastResult.SessionDirectory);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Unable to stop recording", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void TranscribeButton_Click(object sender, RoutedEventArgs e)
    {
        if (_lastResult?.MixedPath is null)
        {
            MessageBox.Show(this, "Record audio before transcribing.", "No audio", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        await RunBusyAsync(async () =>
        {
            TranscriptStateText.Text = "Diarizing";
            TranscriptBox.Text = "Waiting for OpenAI speaker diarization...";
            MinutesBox.Text = "Waiting for transcript...";
            var transcriptProgress = new Progress<string>(message => TranscriptStateText.Text = message);
            var transcript = await _transcription.TranscribeAsync(_lastResult.MixedPath, _settings.OpenAiKey, transcriptProgress);
            TranscriptBox.Text = string.IsNullOrWhiteSpace(transcript) ? "(No text returned.)" : transcript.Trim();
            _lastResult = MeetingArtifactStore.SaveTranscript(_lastResult, transcript);
            RenderResult(_lastResult);

            TranscriptStateText.Text = "Writing minutes";
            MinutesBox.Text = "Generating meeting minutes...";
            var minutesProgress = new Progress<string>(message => TranscriptStateText.Text = message);
            var minutes = await _minutes.CreateMinutesAsync(transcript, _settings.OpenAiKey, _settings.NotesModel, minutesProgress);
            _lastResult = MeetingArtifactStore.SaveMinutes(_lastResult, minutes);
            MinutesBox.Text = minutes.Trim();
            RenderResult(_lastResult);
            RefreshLibrary(_lastResult.SessionDirectory);
            TranscriptStateText.Text = "Done";
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

    private async void LibraryCreateNotesButton_Click(object sender, RoutedEventArgs e)
    {
        var item = SelectedLibraryMeeting();
        if (item?.RecordingPath is null || !File.Exists(item.RecordingPath))
        {
            MessageBox.Show(this, "Select a saved meeting with a recording first.", "No recording", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var refreshDirectory = item.DirectoryPath;
        await RunBusyAsync(async () =>
        {
            SetLibraryButtons(false, false, false, false, false);
            LibraryTranscriptBox.Text = "Waiting for OpenAI speaker diarization...";
            LibraryMinutesBox.Text = "Waiting for transcript...";

            var result = RecordingResultFromLibraryItem(item);
            var transcriptProgress = new Progress<string>(message => LibraryDetailsText.Text = BuildLibraryProcessingText(result, message));
            var transcript = await _transcription.TranscribeAsync(result.MixedPath!, _settings.OpenAiKey, transcriptProgress);
            result = MeetingArtifactStore.SaveTranscript(result, transcript);
            LibraryTranscriptBox.Text = string.IsNullOrWhiteSpace(transcript) ? "(No text returned.)" : transcript.Trim();

            LibraryMinutesBox.Text = "Generating meeting minutes...";
            var minutesProgress = new Progress<string>(message => LibraryDetailsText.Text = BuildLibraryProcessingText(result, message));
            var minutes = await _minutes.CreateMinutesAsync(transcript, _settings.OpenAiKey, _settings.NotesModel, minutesProgress);
            result = MeetingArtifactStore.SaveMinutes(result, minutes);
            LibraryMinutesBox.Text = minutes.Trim();

            _lastResult = result;
            RenderResult(_lastResult);
            refreshDirectory = result.SessionDirectory;
            LibraryDetailsText.Text = BuildLibraryProcessingText(result, "Done");
        });
        RefreshLibrary(refreshDirectory);
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
            TranscribeButton.IsEnabled = false;
            OpenFolderButton.IsEnabled = false;
            return;
        }

        ResultText.Text =
            $"{result.Name}\n" +
            $"{result.Duration.TotalSeconds:F1}s\n\n" +
            $"Recording: {FormatMetrics(result.Mixed)}\n" +
            $"Transcript: {SavedState(result.TranscriptPath)}\n" +
            $"Minutes: {SavedState(result.MinutesPath)}\n\n" +
            $"Folder: {result.SessionDirectory}";
        TranscribeButton.IsEnabled = !_isBusy && result.MixedPath is not null && File.Exists(result.MixedPath);
        OpenFolderButton.IsEnabled = !_isBusy && Directory.Exists(result.SessionDirectory);
        OpenRecordingButton.IsEnabled = !_isBusy && result.MixedPath is not null && File.Exists(result.MixedPath);
        OpenTranscriptButton.IsEnabled = !_isBusy && result.TranscriptPath is not null && File.Exists(result.TranscriptPath);
        OpenMinutesButton.IsEnabled = !_isBusy && result.MinutesPath is not null && File.Exists(result.MinutesPath);
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

    private async Task RunBusyAsync(Func<Task> action)
    {
        if (_isBusy)
        {
            return;
        }

        _isBusy = true;
        ToggleButtons(false);
        try
        {
            await action();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "ADsum", MessageBoxButton.OK, MessageBoxImage.Error);
            TranscriptStateText.Text = "Ready";
        }
        finally
        {
            _isBusy = false;
            ToggleButtons(!_recorder.IsRecording);
            StopButton.IsEnabled = _recorder.IsRecording;
        }
    }

    private void ToggleButtons(bool enabled)
    {
        RefreshButton.IsEnabled = enabled;
        TestButton.IsEnabled = enabled;
        RecordButton.IsEnabled = enabled;
        SaveKeyButton.IsEnabled = enabled;
        TranscribeButton.IsEnabled = enabled && _lastResult?.MixedPath is not null;
        OpenFolderButton.IsEnabled = _lastResult is not null && Directory.Exists(_lastResult.SessionDirectory);
        OpenRecordingButton.IsEnabled = enabled && _lastResult?.MixedPath is not null && File.Exists(_lastResult.MixedPath);
        OpenTranscriptButton.IsEnabled = enabled && _lastResult?.TranscriptPath is not null && File.Exists(_lastResult.TranscriptPath);
        OpenMinutesButton.IsEnabled = enabled && _lastResult?.MinutesPath is not null && File.Exists(_lastResult.MinutesPath);
        LibraryRefreshButton.IsEnabled = enabled;
        LibraryList.IsEnabled = enabled;
        if (enabled)
        {
            RenderLibrarySelection(SelectedLibraryMeeting());
        }
        else
        {
            SetLibraryButtons(false, false, false, false, false);
        }
    }

    private void ClearReviewNotes()
    {
        TranscriptStateText.Text = "Ready";
        TranscriptBox.Text = "Record audio, then create notes.";
        MinutesBox.Text = "Record audio, then create notes.";
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
            SetLibraryButtons(false, false, false, false, false);
            return;
        }

        LibraryTitleText.Text = item.Topic;
        LibraryDetailsText.Text =
            $"{item.DateText}\n" +
            $"{item.FileSummary}\n" +
            $"{item.DirectoryPath}";
        LibraryMinutesBox.Text = ReadTextPreview(item.MinutesPath, "No meeting minutes saved for this meeting.");
        LibraryTranscriptBox.Text = ReadTextPreview(item.TranscriptPath, "No transcript saved for this meeting.");
        SetLibraryButtons(
            Directory.Exists(item.DirectoryPath),
            item.RecordingPath is not null && File.Exists(item.RecordingPath),
            item.RecordingPath is not null && File.Exists(item.RecordingPath),
            item.MinutesPath is not null && File.Exists(item.MinutesPath),
            item.TranscriptPath is not null && File.Exists(item.TranscriptPath));
    }

    private MeetingLibraryItem? SelectedLibraryMeeting() => (MeetingLibraryItem?)LibraryList.SelectedItem;

    private void SetLibraryButtons(bool folder, bool recording, bool createNotes, bool minutes, bool transcript)
    {
        LibraryOpenFolderButton.IsEnabled = !_isBusy && folder;
        LibraryOpenRecordingButton.IsEnabled = !_isBusy && recording;
        LibraryCreateNotesButton.IsEnabled = !_isBusy && createNotes;
        LibraryOpenMinutesButton.IsEnabled = !_isBusy && minutes;
        LibraryOpenTranscriptButton.IsEnabled = !_isBusy && transcript;
    }

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

    private static RecordingResult RecordingResultFromLibraryItem(MeetingLibraryItem item)
    {
        if (item.RecordingPath is null || !File.Exists(item.RecordingPath))
        {
            throw new FileNotFoundException("Saved recording was not found.", item.RecordingPath);
        }

        var mixedMetrics = MeetingRecorder.MeasureWaveFile(item.RecordingPath);
        return new RecordingResult(
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
    }

    private static string BuildLibraryProcessingText(RecordingResult result, string state)
    {
        return
            $"{result.StartedAt:yyyy-MM-dd HH:mm}\n" +
            $"{FormatMetrics(result.Mixed)}\n" +
            $"{result.SessionDirectory}\n\n" +
            state;
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
}
