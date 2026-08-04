using System.IO;
using System.Runtime.InteropServices;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace ADsum.Desktop.Services;

public sealed class MeetingRecorder : IDisposable
{
    private const int MixedSampleRate = 16000;
    private const float TargetActiveRms = 0.12f;
    private const float MinimumActiveRms = 0.002f;
    private const float MinimumPeakForGain = 0.004f;
    private const float ActiveThresholdFloor = 0.003f;
    private const float ActivePeakFraction = 0.08f;
    private const float MinimumTrackGain = 0.5f;
    private const float MaximumTrackGain = 10.0f;
    private const float MaximumNormalizedPeak = 0.95f;
    private const int StreamingFrameBufferSize = 4096;
    private const int StreamingMixBufferSize = 16384;
    private static readonly TimeSpan TimelineGapTolerance = TimeSpan.FromMilliseconds(40);
    private readonly AudioDeviceService _devices = new();
    private readonly object _micLock = new();
    private readonly object _systemLock = new();
    private WasapiCapture? _microphoneCapture;
    private WasapiLoopbackCapture? _systemCapture;
    private WaveFileWriter? _microphoneWriter;
    private WaveFileWriter? _systemWriter;
    private DateTimeOffset _startedAt;
    private DateTime _startedLocalTime;
    private string _sessionName = "";
    private string _sessionDirectory = "";
    private long _microphoneTimelineBytes;
    private long _systemTimelineBytes;

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

        _sessionName = string.IsNullOrWhiteSpace(name) ? "Untitled meeting" : name.Trim();
        _startedLocalTime = DateTime.Now;
        _sessionDirectory = CreateSessionDirectory(_sessionName, _startedLocalTime);
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
        _microphoneTimelineBytes = 0;
        _systemTimelineBytes = 0;
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
        var mixedPath = Path.Combine(_sessionDirectory, "recording.wav");
        MixWaveFiles(
            new[] { microphonePath, systemPath }.Where(path => path is not null).Cast<string>().ToArray(),
            mixedPath);
        var finalMixedPath = ExistingPath(mixedPath);
        var microphoneMetrics = MeasureWaveFile(microphonePath);
        var systemMetrics = MeasureWaveFile(systemPath);
        var mixedMetrics = MeasureWaveFile(finalMixedPath);
        TryDeleteFile(microphonePath);
        TryDeleteFile(systemPath);

        LastResult = new RecordingResult(
            _sessionName,
            _sessionDirectory,
            _startedLocalTime,
            duration,
            null,
            null,
            finalMixedPath,
            null,
            null,
            microphoneMetrics,
            systemMetrics,
            mixedMetrics);

        MicrophoneLevel = 0;
        SystemLevel = 0;
        return LastResult;
    }

    public async Task<RecordingResult> RunDeviceTestAsync(string name, string microphoneId, string outputId, TimeSpan duration, TimeSpan? toneDelay = null)
    {
        Start(string.IsNullOrWhiteSpace(name) ? "Device test" : name, microphoneId, outputId);
        IDisposable? tone = null;
        try
        {
            var delay = toneDelay.GetValueOrDefault();
            if (delay > TimeSpan.Zero)
            {
                await Task.Delay(delay < duration ? delay : duration);
            }

            if (delay < duration)
            {
                tone = PlayTestTone(outputId);
                await Task.Delay(duration - delay);
            }

            return Stop();
        }
        finally
        {
            tone?.Dispose();
        }
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
            if (_microphoneWriter is not null && _microphoneCapture?.WaveFormat is not null)
            {
                WriteTimelineAligned(_microphoneWriter, _microphoneCapture.WaveFormat, e.Buffer, e.BytesRecorded, ref _microphoneTimelineBytes);
            }
            _microphoneWriter?.Flush();
        }
    }

    private void SystemDataAvailable(object? sender, WaveInEventArgs e)
    {
        SystemLevel = CalculateRms(e.Buffer, e.BytesRecorded, _systemCapture?.WaveFormat);
        lock (_systemLock)
        {
            if (_systemWriter is not null && _systemCapture?.WaveFormat is not null)
            {
                WriteTimelineAligned(_systemWriter, _systemCapture.WaveFormat, e.Buffer, e.BytesRecorded, ref _systemTimelineBytes);
            }
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

    private static string CreateSessionDirectory(string name, DateTime startedAt)
    {
        var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ADsum", "Recordings");
        var stamp = startedAt.ToString("yyyyMMdd-HHmm");
        var slug = MeetingArtifactStore.Slugify(name);
        if (string.IsNullOrWhiteSpace(slug))
        {
            slug = "untitled-meeting";
        }
        return MeetingArtifactStore.UniqueDirectory(root, $"{stamp}-{slug}");
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

    private static void TryDeleteFile(string? path)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Temporary channel files are best-effort cleanup after the mixed recording is written.
        }
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

    private void WriteTimelineAligned(WaveFileWriter writer, WaveFormat format, byte[] buffer, int bytesRecorded, ref long timelineBytes)
    {
        var chunkStartBytes = EstimateChunkStartBytes(format, bytesRecorded);
        var gapBytes = chunkStartBytes - timelineBytes;
        var toleranceBytes = DurationToAlignedBytes(TimelineGapTolerance, format);
        if (gapBytes > toleranceBytes)
        {
            WriteSilence(writer, AlignBytes(gapBytes, format.BlockAlign));
            timelineBytes += AlignBytes(gapBytes, format.BlockAlign);
        }

        writer.Write(buffer, 0, bytesRecorded);
        timelineBytes += bytesRecorded;
    }

    private long EstimateChunkStartBytes(WaveFormat format, int bytesRecorded)
    {
        var averageBytesPerSecond = Math.Max(1, format.AverageBytesPerSecond);
        var chunkDuration = TimeSpan.FromSeconds((double)bytesRecorded / averageBytesPerSecond);
        var chunkStart = DateTimeOffset.Now - _startedAt - chunkDuration;
        if (chunkStart < TimeSpan.Zero)
        {
            chunkStart = TimeSpan.Zero;
        }
        return DurationToAlignedBytes(chunkStart, format);
    }

    private static long DurationToAlignedBytes(TimeSpan duration, WaveFormat format)
    {
        var averageBytesPerSecond = Math.Max(1, format.AverageBytesPerSecond);
        return AlignBytes((long)Math.Round(duration.TotalSeconds * averageBytesPerSecond), format.BlockAlign);
    }

    private static long AlignBytes(long bytes, int blockAlign)
    {
        var alignment = Math.Max(1, blockAlign);
        return bytes - (bytes % alignment);
    }

    private static void WriteSilence(WaveFileWriter writer, long bytes)
    {
        var blockAlign = Math.Max(1, writer.WaveFormat.BlockAlign);
        var bufferLength = AlignBytes(8192, blockAlign);
        var silence = new byte[Math.Max(blockAlign, bufferLength)];
        while (bytes > 0)
        {
            var count = (int)Math.Min(silence.Length, bytes);
            writer.Write(silence, 0, count);
            bytes -= count;
        }
    }

    public static void MixWaveFiles(IReadOnlyList<string> paths, string outputPath)
    {
        var tracks = paths
            .Where(path => File.Exists(path))
            .Select(CreateStreamingTrack)
            .Where(track => track is not null)
            .Cast<StreamingTrack>()
            .Select(AnalyzeSpeechTrack)
            .ToList();

        if (tracks.Count == 0)
        {
            return;
        }

        var outputFullPath = Path.GetFullPath(outputPath);
        var outputDirectory = Path.GetDirectoryName(outputFullPath)
            ?? throw new InvalidOperationException("The mixed recording needs a parent directory.");
        Directory.CreateDirectory(outputDirectory);
        DeleteStaleMixFiles(outputDirectory, Path.GetFileName(outputFullPath));

        var temporaryStem = $".{Path.GetFileName(outputFullPath)}.{Guid.NewGuid():N}";
        var floatMixPath = Path.Combine(outputDirectory, $"{temporaryStem}.float-mix.tmp");
        var pendingWavePath = Path.Combine(outputDirectory, $"{temporaryStem}.pending-wave.tmp");

        try
        {
            var mixedSampleCount = tracks.Max(track => track.TargetSampleCount);
            var mixedPeak = WriteUnscaledFloatMix(tracks, floatMixPath, mixedSampleCount);
            WriteFinalPcm16(floatMixPath, pendingWavePath, mixedSampleCount, mixedPeak);
            File.Move(pendingWavePath, outputFullPath, true);
        }
        finally
        {
            TryDeleteFile(floatMixPath);
            TryDeleteFile(pendingWavePath);
        }
    }

    private static void DeleteStaleMixFiles(string outputDirectory, string outputFileName)
    {
        try
        {
            var prefix = $".{outputFileName}.";
            foreach (var path in Directory.EnumerateFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly))
            {
                var name = Path.GetFileName(path);
                if (!name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ||
                    (!name.EndsWith(".float-mix.tmp", StringComparison.OrdinalIgnoreCase) &&
                     !name.EndsWith(".pending-wave.tmp", StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                // A previous crash can leave hundreds of megabytes behind for
                // a long meeting. ADsum permits only one recording-capable
                // process, so no live mixer can own this output path here.
                TryDeleteFile(path);
            }
        }
        catch
        {
            // Stale-file cleanup is best effort and must not prevent Stop from
            // preserving the newly recorded meeting.
        }
    }

    private static StreamingTrack? CreateStreamingTrack(string path)
    {
        using var reader = new AudioFileReader(path);
        var channels = reader.WaveFormat.Channels;
        var sampleRate = reader.WaveFormat.SampleRate;
        var blockAlign = reader.WaveFormat.BlockAlign;
        if (channels <= 0 || sampleRate <= 0 || blockAlign <= 0)
        {
            return null;
        }

        var sourceSampleCount = reader.Length / blockAlign;
        if (sourceSampleCount <= 0)
        {
            return null;
        }

        long targetSampleCount;
        if (sampleRate == MixedSampleRate)
        {
            targetSampleCount = sourceSampleCount;
        }
        else
        {
            var duration = (double)sourceSampleCount / sampleRate;
            targetSampleCount = Math.Max(1L, checked((long)Math.Round(duration * MixedSampleRate)));
        }

        return new StreamingTrack(path, sampleRate, sourceSampleCount, targetSampleCount, 1.0f);
    }

    private static StreamingTrack AnalyzeSpeechTrack(StreamingTrack track)
    {
        var peak = 0f;
        using (var samples = new ResampledSampleReader(track))
        {
            while (samples.TryRead(out var sample))
            {
                peak = Math.Max(peak, Math.Abs(sample));
            }
        }

        if (peak < MinimumPeakForGain)
        {
            return track;
        }

        var activeThreshold = Math.Max(ActiveThresholdFloor, peak * ActivePeakFraction);
        double activeSum = 0;
        long activeSamples = 0;
        using (var samples = new ResampledSampleReader(track))
        {
            while (samples.TryRead(out var sample))
            {
                if (Math.Abs(sample) < activeThreshold)
                {
                    continue;
                }

                activeSum += sample * sample;
                activeSamples++;
            }
        }

        if (activeSamples == 0)
        {
            return track;
        }

        var activeRms = (float)Math.Sqrt(activeSum / activeSamples);
        if (activeRms < MinimumActiveRms)
        {
            return track;
        }

        var gain = Math.Clamp(TargetActiveRms / activeRms, MinimumTrackGain, MaximumTrackGain);
        gain = Math.Min(gain, MaximumNormalizedPeak / peak);
        if (Math.Abs(gain - 1.0f) < 0.01f)
        {
            gain = 1.0f;
        }

        return track with { Gain = gain };
    }

    private static float WriteUnscaledFloatMix(
        IReadOnlyList<StreamingTrack> tracks,
        string floatMixPath,
        long mixedSampleCount)
    {
        var readers = tracks.Select(track => new ResampledSampleReader(track)).ToList();
        var buffer = new float[StreamingMixBufferSize];
        var buffered = 0;
        var peak = 0f;

        try
        {
            using var output = new FileStream(
                floatMixPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                StreamingMixBufferSize * sizeof(float),
                FileOptions.SequentialScan);

            for (long index = 0; index < mixedSampleCount; index++)
            {
                var mixedSample = 0f;
                for (var trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
                {
                    if (readers[trackIndex].TryRead(out var sample))
                    {
                        var normalizedSample = sample * tracks[trackIndex].Gain;
                        mixedSample += normalizedSample / tracks.Count;
                    }
                }

                peak = Math.Max(peak, Math.Abs(mixedSample));
                buffer[buffered++] = mixedSample;
                if (buffered == buffer.Length)
                {
                    WriteFloatBuffer(output, buffer, buffered);
                    buffered = 0;
                }
            }

            if (buffered > 0)
            {
                WriteFloatBuffer(output, buffer, buffered);
            }
        }
        finally
        {
            foreach (var reader in readers)
            {
                reader.Dispose();
            }
        }

        return peak;
    }

    private static void WriteFloatBuffer(Stream output, float[] buffer, int sampleCount)
    {
        output.Write(MemoryMarshal.AsBytes(buffer.AsSpan(0, sampleCount)));
    }

    private static void WriteFinalPcm16(
        string floatMixPath,
        string pendingWavePath,
        long sampleCount,
        float mixedPeak)
    {
        var scale = mixedPeak > 0.98f ? 0.98f / mixedPeak : 1.0f;
        var floatBytes = new byte[StreamingMixBufferSize * sizeof(float)];
        var pcmBytes = new byte[StreamingMixBufferSize * sizeof(short)];

        using var input = new FileStream(
            floatMixPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            floatBytes.Length,
            FileOptions.SequentialScan);
        using var writer = new WaveFileWriter(pendingWavePath, new WaveFormat(MixedSampleRate, 16, 1));

        var remaining = sampleCount;
        while (remaining > 0)
        {
            var samplesThisPass = (int)Math.Min(StreamingMixBufferSize, remaining);
            var bytesThisPass = samplesThisPass * sizeof(float);
            input.ReadExactly(floatBytes.AsSpan(0, bytesThisPass));

            var floatSamples = MemoryMarshal.Cast<byte, float>(floatBytes.AsSpan(0, bytesThisPass));
            var pcmSamples = MemoryMarshal.Cast<byte, short>(pcmBytes.AsSpan(0, samplesThisPass * sizeof(short)));
            for (var index = 0; index < samplesThisPass; index++)
            {
                var sample = floatSamples[index];
                if (mixedPeak > 0.98f)
                {
                    sample *= scale;
                }

                pcmSamples[index] = (short)Math.Clamp(sample * 32767, short.MinValue, short.MaxValue);
            }

            writer.Write(pcmBytes, 0, samplesThisPass * sizeof(short));
            remaining -= samplesThisPass;
        }
    }

    public static TrackMetrics MeasureWaveFile(string? path)
    {
        if (path is null || !File.Exists(path))
        {
            return new TrackMetrics(path, TimeSpan.Zero, 0, 0);
        }

        using var track = new MonoSampleReader(path);
        if (track.SampleRate <= 0)
        {
            return new TrackMetrics(path, TimeSpan.Zero, 0, 0);
        }

        long sampleCount = 0;
        double sumSquares = 0;
        var peak = 0f;
        while (track.TryRead(out var sample))
        {
            peak = Math.Max(peak, Math.Abs(sample));
            sumSquares += sample * sample;
            sampleCount++;
        }

        if (sampleCount == 0)
        {
            return new TrackMetrics(path, TimeSpan.Zero, 0, 0);
        }

        var rms = (float)Math.Sqrt(sumSquares / sampleCount);
        return new TrackMetrics(
            path,
            TimeSpan.FromSeconds((double)sampleCount / track.SampleRate),
            peak,
            rms);
    }

    private sealed class MonoSampleReader : IDisposable
    {
        private readonly AudioFileReader _reader;
        private readonly int _channels;
        private readonly float[] _buffer;
        private int _bufferOffset;
        private int _bufferCount;

        public MonoSampleReader(string path)
        {
            _reader = new AudioFileReader(path);
            _channels = _reader.WaveFormat.Channels;
            SampleRate = _reader.WaveFormat.SampleRate;
            _buffer = new float[Math.Max(_channels, StreamingFrameBufferSize * _channels)];
        }

        public int SampleRate { get; }

        public bool TryRead(out float monoSample)
        {
            while (_bufferCount - _bufferOffset < _channels)
            {
                var remaining = _bufferCount - _bufferOffset;
                if (remaining > 0)
                {
                    Array.Copy(_buffer, _bufferOffset, _buffer, 0, remaining);
                }

                _bufferOffset = 0;
                _bufferCount = remaining;
                var read = _reader.Read(_buffer, remaining, _buffer.Length - remaining);
                if (read == 0)
                {
                    monoSample = 0;
                    return false;
                }

                _bufferCount += read;
            }

            var sum = 0f;
            for (var channel = 0; channel < _channels; channel++)
            {
                sum += _buffer[_bufferOffset + channel];
            }

            _bufferOffset += _channels;
            monoSample = sum / _channels;
            return true;
        }

        public void Dispose()
        {
            _reader.Dispose();
        }
    }

    private sealed class ResampledSampleReader : IDisposable
    {
        private readonly StreamingTrack _track;
        private readonly MonoSampleReader _source;
        private long _targetIndex;
        private long _sourceIndex;
        private float _leftSample;
        private float _rightSample;
        private bool _hasSamples;

        public ResampledSampleReader(StreamingTrack track)
        {
            _track = track;
            _source = new MonoSampleReader(track.Path);
            _hasSamples = _source.TryRead(out _leftSample);
            if (_hasSamples)
            {
                _rightSample = _source.TryRead(out var right) ? right : _leftSample;
            }
        }

        public bool TryRead(out float sample)
        {
            if (!_hasSamples || _targetIndex >= _track.TargetSampleCount)
            {
                sample = 0;
                return false;
            }

            if (_track.SampleRate == MixedSampleRate)
            {
                sample = _leftSample;
                AdvanceSource();
                _targetIndex++;
                return true;
            }

            var sourcePosition = (double)_targetIndex / MixedSampleRate * _track.SampleRate;
            var leftIndex = (long)Math.Floor(sourcePosition);
            while (_sourceIndex < leftIndex && _sourceIndex < _track.SourceSampleCount - 1)
            {
                AdvanceSource();
            }

            var blend = sourcePosition - leftIndex;
            sample = (float)((_leftSample * (1 - blend)) + (_rightSample * blend));
            _targetIndex++;
            return true;
        }

        public void Dispose()
        {
            _source.Dispose();
        }

        private void AdvanceSource()
        {
            if (_sourceIndex >= _track.SourceSampleCount - 1)
            {
                _rightSample = _leftSample;
                return;
            }

            _leftSample = _rightSample;
            _sourceIndex++;
            _rightSample = _source.TryRead(out var next) ? next : _leftSample;
        }
    }

    private sealed record StreamingTrack(
        string Path,
        int SampleRate,
        long SourceSampleCount,
        long TargetSampleCount,
        float Gain);
}
