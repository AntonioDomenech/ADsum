using System.IO;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace ADsum.Desktop.Services;

public sealed class MeetingRecorder : IDisposable
{
    private const int MixedSampleRate = 16000;
    private readonly AudioDeviceService _devices = new();
    private readonly object _micLock = new();
    private readonly object _systemLock = new();
    private WasapiCapture? _microphoneCapture;
    private WasapiLoopbackCapture? _systemCapture;
    private WaveFileWriter? _microphoneWriter;
    private WaveFileWriter? _systemWriter;
    private DateTimeOffset _startedAt;
    private string _sessionName = "";
    private string _sessionDirectory = "";

    public bool IsRecording { get; private set; }

    public float MicrophoneLevel { get; private set; }

    public float SystemLevel { get; private set; }

    public TimeSpan Elapsed => IsRecording ? DateTimeOffset.Now - _startedAt : TimeSpan.Zero;

    public RecordingResult? LastResult { get; private set; }

    public void Start(string name, string microphoneId, string outputId)
    {
        if (IsRecording)
        {
            throw new InvalidOperationException("A recording is already active.");
        }

        _sessionName = string.IsNullOrWhiteSpace(name) ? $"Meeting {DateTime.Now:yyyy-MM-dd HH.mm}" : name.Trim();
        _sessionDirectory = CreateSessionDirectory(_sessionName);
        Directory.CreateDirectory(_sessionDirectory);

        var microphone = _devices.GetMicrophone(microphoneId);
        var output = _devices.GetRenderDevice(outputId);
        var microphonePath = Path.Combine(_sessionDirectory, "microphone.wav");
        var systemPath = Path.Combine(_sessionDirectory, "system.wav");

        _microphoneCapture = new WasapiCapture(microphone);
        _systemCapture = new WasapiLoopbackCapture(output);
        _microphoneWriter = new WaveFileWriter(microphonePath, _microphoneCapture.WaveFormat);
        _systemWriter = new WaveFileWriter(systemPath, _systemCapture.WaveFormat);

        _microphoneCapture.DataAvailable += MicrophoneDataAvailable;
        _systemCapture.DataAvailable += SystemDataAvailable;
        _microphoneCapture.RecordingStopped += CaptureStopped;
        _systemCapture.RecordingStopped += CaptureStopped;

        _startedAt = DateTimeOffset.Now;
        MicrophoneLevel = 0;
        SystemLevel = 0;
        IsRecording = true;

        try
        {
            _microphoneCapture.StartRecording();
            _systemCapture.StartRecording();
        }
        catch
        {
            StopCaptureObjects();
            IsRecording = false;
            throw;
        }
    }

    public RecordingResult Stop()
    {
        if (!IsRecording)
        {
            throw new InvalidOperationException("No recording is active.");
        }

        var duration = DateTimeOffset.Now - _startedAt;
        IsRecording = false;
        StopCaptureObjects();

        var microphonePath = ExistingPath(Path.Combine(_sessionDirectory, "microphone.wav"));
        var systemPath = ExistingPath(Path.Combine(_sessionDirectory, "system.wav"));
        var mixedPath = Path.Combine(_sessionDirectory, "mixed.wav");
        MixWaveFiles(
            new[] { microphonePath, systemPath }.Where(path => path is not null).Cast<string>().ToArray(),
            mixedPath);
        var finalMixedPath = ExistingPath(mixedPath);

        LastResult = new RecordingResult(
            _sessionName,
            _sessionDirectory,
            duration,
            microphonePath,
            systemPath,
            finalMixedPath,
            MeasureWaveFile(microphonePath),
            MeasureWaveFile(systemPath),
            MeasureWaveFile(finalMixedPath));

        MicrophoneLevel = 0;
        SystemLevel = 0;
        return LastResult;
    }

    public async Task<RecordingResult> RunDeviceTestAsync(string name, string microphoneId, string outputId, TimeSpan duration)
    {
        Start(string.IsNullOrWhiteSpace(name) ? "Device test" : name, microphoneId, outputId);
        using var tone = PlayTestTone(outputId);
        await Task.Delay(duration);
        tone?.Dispose();
        return Stop();
    }

    public void Dispose()
    {
        if (IsRecording)
        {
            Stop();
        }
        StopCaptureObjects();
    }

    private void MicrophoneDataAvailable(object? sender, WaveInEventArgs e)
    {
        MicrophoneLevel = CalculateRms(e.Buffer, e.BytesRecorded, _microphoneCapture?.WaveFormat);
        lock (_micLock)
        {
            _microphoneWriter?.Write(e.Buffer, 0, e.BytesRecorded);
            _microphoneWriter?.Flush();
        }
    }

    private void SystemDataAvailable(object? sender, WaveInEventArgs e)
    {
        SystemLevel = CalculateRms(e.Buffer, e.BytesRecorded, _systemCapture?.WaveFormat);
        lock (_systemLock)
        {
            _systemWriter?.Write(e.Buffer, 0, e.BytesRecorded);
            _systemWriter?.Flush();
        }
    }

    private static void CaptureStopped(object? sender, StoppedEventArgs e)
    {
        if (e.Exception is not null)
        {
            Console.Error.WriteLine(e.Exception);
        }
    }

    private void StopCaptureObjects()
    {
        var microphoneCapture = _microphoneCapture;
        var systemCapture = _systemCapture;
        _microphoneCapture = null;
        _systemCapture = null;

        if (microphoneCapture is not null)
        {
            microphoneCapture.DataAvailable -= MicrophoneDataAvailable;
            microphoneCapture.RecordingStopped -= CaptureStopped;
            SafeStop(microphoneCapture);
            microphoneCapture.Dispose();
        }

        if (systemCapture is not null)
        {
            systemCapture.DataAvailable -= SystemDataAvailable;
            systemCapture.RecordingStopped -= CaptureStopped;
            SafeStop(systemCapture);
            systemCapture.Dispose();
        }

        lock (_micLock)
        {
            _microphoneWriter?.Dispose();
            _microphoneWriter = null;
        }

        lock (_systemLock)
        {
            _systemWriter?.Dispose();
            _systemWriter = null;
        }
    }

    private static void SafeStop(IWaveIn capture)
    {
        try
        {
            capture.StopRecording();
        }
        catch
        {
            // Device may already be stopped or disconnected.
        }
    }

    private IDisposable? PlayTestTone(string outputId)
    {
        try
        {
            var output = _devices.GetRenderDevice(outputId);
            var signal = new SignalGenerator(output.AudioClient.MixFormat.SampleRate, output.AudioClient.MixFormat.Channels)
            {
                Gain = 0.12,
                Frequency = 660,
                Type = SignalGeneratorType.Sin
            };
            var player = new WasapiOut(output, AudioClientShareMode.Shared, false, 100);
            player.Init(signal);
            player.Play();
            return player;
        }
        catch
        {
            return null;
        }
    }

    private static string CreateSessionDirectory(string name)
    {
        var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ADsum", "Recordings");
        var stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        var slug = string.Join("-", name.Split(Path.GetInvalidFileNameChars(), StringSplitOptions.RemoveEmptyEntries))
            .Trim()
            .Replace(" ", "-")
            .ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(slug))
        {
            slug = "session";
        }
        return Path.Combine(root, $"{stamp}-{slug}");
    }

    private static string? ExistingPath(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }
        var info = new FileInfo(path);
        return info.Length > 44 ? path : null;
    }

    private static float CalculateRms(byte[] buffer, int bytesRecorded, WaveFormat? format)
    {
        if (format is null || bytesRecorded <= 0)
        {
            return 0;
        }

        double sum = 0;
        var samples = 0;
        if (format.Encoding == WaveFormatEncoding.IeeeFloat && format.BitsPerSample == 32)
        {
            for (var offset = 0; offset + 4 <= bytesRecorded; offset += 4)
            {
                var sample = BitConverter.ToSingle(buffer, offset);
                sum += sample * sample;
                samples++;
            }
        }
        else if (format.BitsPerSample == 16)
        {
            for (var offset = 0; offset + 2 <= bytesRecorded; offset += 2)
            {
                var sample = BitConverter.ToInt16(buffer, offset) / 32768.0;
                sum += sample * sample;
                samples++;
            }
        }

        return samples == 0 ? 0 : (float)Math.Sqrt(sum / samples);
    }

    public static void MixWaveFiles(IReadOnlyList<string> paths, string outputPath)
    {
        var tracks = paths
            .Where(path => File.Exists(path))
            .Select(ReadMonoSamples)
            .Where(track => track.Samples.Length > 0)
            .Select(track => Resample(track.Samples, track.SampleRate, MixedSampleRate))
            .ToList();

        if (tracks.Count == 0)
        {
            return;
        }

        var length = tracks.Max(track => track.Length);
        var mixed = new float[length];
        foreach (var track in tracks)
        {
            for (var index = 0; index < track.Length; index++)
            {
                mixed[index] += track[index] / tracks.Count;
            }
        }

        var peak = mixed.Select(Math.Abs).DefaultIfEmpty(0).Max();
        if (peak > 0.98f)
        {
            var scale = 0.98f / peak;
            for (var index = 0; index < mixed.Length; index++)
            {
                mixed[index] *= scale;
            }
        }

        WritePcm16(outputPath, mixed, MixedSampleRate);
    }

    public static TrackMetrics MeasureWaveFile(string? path)
    {
        if (path is null || !File.Exists(path))
        {
            return new TrackMetrics(path, TimeSpan.Zero, 0, 0);
        }

        var track = ReadMonoSamples(path);
        if (track.Samples.Length == 0 || track.SampleRate <= 0)
        {
            return new TrackMetrics(path, TimeSpan.Zero, 0, 0);
        }

        var peak = track.Samples.Select(Math.Abs).DefaultIfEmpty(0).Max();
        var rms = (float)Math.Sqrt(track.Samples.Select(sample => sample * sample).DefaultIfEmpty(0).Average());
        return new TrackMetrics(path, TimeSpan.FromSeconds((double)track.Samples.Length / track.SampleRate), peak, rms);
    }

    private static AudioTrack ReadMonoSamples(string path)
    {
        using var reader = new AudioFileReader(path);
        var channels = reader.WaveFormat.Channels;
        var sampleRate = reader.WaveFormat.SampleRate;
        var buffer = new float[sampleRate * channels];
        var samples = new List<float>();

        int read;
        while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            var frames = read / channels;
            for (var frame = 0; frame < frames; frame++)
            {
                var sum = 0f;
                for (var channel = 0; channel < channels; channel++)
                {
                    sum += buffer[(frame * channels) + channel];
                }
                samples.Add(sum / channels);
            }
        }

        return new AudioTrack(samples.ToArray(), sampleRate);
    }

    private static float[] Resample(float[] samples, int sourceRate, int targetRate)
    {
        if (samples.Length == 0 || sourceRate == targetRate)
        {
            return samples;
        }

        var duration = (double)samples.Length / sourceRate;
        var targetLength = Math.Max(1, (int)Math.Round(duration * targetRate));
        var output = new float[targetLength];
        for (var index = 0; index < targetLength; index++)
        {
            var sourcePosition = (double)index / targetRate * sourceRate;
            var left = (int)Math.Floor(sourcePosition);
            var right = Math.Min(left + 1, samples.Length - 1);
            var blend = sourcePosition - left;
            output[index] = (float)((samples[left] * (1 - blend)) + (samples[right] * blend));
        }
        return output;
    }

    private static void WritePcm16(string outputPath, float[] samples, int sampleRate)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        using var writer = new WaveFileWriter(outputPath, new WaveFormat(sampleRate, 16, 1));
        var bytes = new byte[samples.Length * 2];
        for (var index = 0; index < samples.Length; index++)
        {
            var value = (short)Math.Clamp(samples[index] * 32767, short.MinValue, short.MaxValue);
            BitConverter.GetBytes(value).CopyTo(bytes, index * 2);
        }
        writer.Write(bytes, 0, bytes.Length);
    }

    private sealed record AudioTrack(float[] Samples, int SampleRate);
}
