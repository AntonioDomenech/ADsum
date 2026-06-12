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
    private readonly DispatcherTimer _timer;
    private RecordingResult? _lastResult;

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
        SessionNameBox.Text = $"Meeting {DateTime.Now:yyyy-MM-dd HH.mm}";
        KeyStateText.Text = _settings.HasOpenAiKey ? "OpenAI key configured" : "OpenAI key not configured";
        RefreshDevices();
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
            StateText.Text = "Recording";
            StopButton.IsEnabled = true;
            RecordButton.IsEnabled = false;
            TestButton.IsEnabled = false;
            RefreshButton.IsEnabled = false;
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
            TranscriptStateText.Text = "Transcribing";
            TranscriptBox.Text = "Waiting for OpenAI...";
            var text = await _transcription.TranscribeAsync(_lastResult.MixedPath, _settings.OpenAiKey);
            TranscriptBox.Text = string.IsNullOrWhiteSpace(text) ? "(No text returned.)" : text.Trim();
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
            return;
        }

        ResultText.Text =
            $"{result.Name}\n" +
            $"{result.Duration.TotalSeconds:F1}s\n\n" +
            $"Mic: {FormatMetrics(result.Microphone)}\n" +
            $"System: {FormatMetrics(result.System)}\n" +
            $"Mixed: {FormatMetrics(result.Mixed)}\n\n" +
            $"Folder: {result.SessionDirectory}";
        TranscribeButton.IsEnabled = result.MixedPath is not null && File.Exists(result.MixedPath);
    }

    private static string FormatMetrics(TrackMetrics metrics)
    {
        if (metrics.Path is null)
        {
            return "missing";
        }
        return $"{metrics.Duration.TotalSeconds:F1}s, peak {metrics.Peak:F3}, rms {metrics.Rms:F3}";
    }

    private string SelectedMicrophoneId() => ((AudioDeviceInfo?)MicrophoneCombo.SelectedItem)?.Id ?? string.Empty;

    private string SelectedOutputId() => ((AudioDeviceInfo?)OutputCombo.SelectedItem)?.Id ?? string.Empty;

    private void UpdateWarnings()
    {
        MicrophoneWarningText.Text = ((AudioDeviceInfo?)MicrophoneCombo.SelectedItem)?.Warning ?? "";
        OutputWarningText.Text = ((AudioDeviceInfo?)OutputCombo.SelectedItem)?.Warning ?? "";
    }

    private async Task RunBusyAsync(Func<Task> action)
    {
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
    }
}
