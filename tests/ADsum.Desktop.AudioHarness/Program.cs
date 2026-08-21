using System.Diagnostics;
using ADsum.Desktop.Services;
using NAudio.Wave;

const int mixedSampleRate = 16000;
const float targetActiveRms = 0.12f;
const float minimumActiveRms = 0.002f;
const float minimumPeakForGain = 0.004f;
const float activeThresholdFloor = 0.003f;
const float activePeakFraction = 0.08f;
const float minimumTrackGain = 0.5f;
const float maximumTrackGain = 10.0f;
const float maximumNormalizedPeak = 0.95f;

var root = Path.Combine(Path.GetTempPath(), $"adsum-audio-harness-{Guid.NewGuid():N}");
Directory.CreateDirectory(root);

try
{
    if (args.Length == 2 && args[0].Equals("--memory-minutes", StringComparison.OrdinalIgnoreCase))
    {
        RunMemoryCheck(root, int.Parse(args[1]));
        return;
    }

    RunCompatibilityChecks(root);
    RunLibraryDurationChecks(root);
    RunCompressionAndTranscriptVersionChecks(root);
    RunDiarizedAlignmentChecks();
}

finally
{
    Directory.Delete(root, true);
}

void RunLibraryDurationChecks(string directory)
{
    var libraryRoot = Path.Combine(directory, "library");
    var meetingDirectory = Path.Combine(libraryRoot, "20260804-1123-long-meeting");
    Directory.CreateDirectory(meetingDirectory);
    var recordingPath = Path.Combine(meetingDirectory, "recording-long-meeting.wav");
    var expectedDuration = new TimeSpan(1, 30, 17);
    WriteSparseSilentPcm16Wave(recordingPath, mixedSampleRate, expectedDuration);

    var meetings = new MeetingLibraryService(libraryRoot).GetMeetings();
    Assert(meetings.Count == 1, "library did not find the recording fixture");
    Assert(meetings[0].RecordingDuration == expectedDuration, "library recording duration changed");
    Assert(meetings[0].DurationText == "Duration: 1:30:17", "library duration text is incorrect");

    Console.WriteLine($"PASS library_duration={meetings[0].DurationText}");
}

void RunCompressionAndTranscriptVersionChecks(string directory)
{
    var libraryRoot = Path.Combine(directory, "compressed-library");
    var meetingDirectory = Path.Combine(libraryRoot, "20260821-0900-compression-and-versions");
    Directory.CreateDirectory(meetingDirectory);
    var originalPath = Path.Combine(meetingDirectory, MeetingArtifactStore.RecordingFileName);
    WritePcm16Fixture(originalPath, 16000, 1, 4.0, (frame, _, rate) =>
        (float)(0.12 * Math.Sin(2 * Math.PI * 220 * frame / rate)));

    var compression = new AudioCompressionService();
    var compressedPath = compression.EnsureCompressedAsync(originalPath).GetAwaiter().GetResult();
    Assert(File.Exists(originalPath), "compression removed the original recording");
    Assert(File.Exists(compressedPath), "compression did not create the MP3");
    Assert(Path.GetFileName(compressedPath) == AudioCompressionService.CompressedFileName,
        "compressed MP3 used an unstable filename");
    using (var mp3 = new Mp3FileReader(compressedPath))
    {
        Assert(Math.Abs((mp3.TotalTime - TimeSpan.FromSeconds(4)).TotalMilliseconds) < 150,
            "compressed MP3 duration changed materially");
    }

    var firstLength = new FileInfo(compressedPath).Length;
    var reusedPath = compression.EnsureCompressedAsync(originalPath).GetAwaiter().GetResult();
    Assert(reusedPath == compressedPath, "compression did not reuse the stable MP3 path");
    Assert(new FileInfo(compressedPath).Length == firstLength, "reusing the MP3 unexpectedly rewrote its content");
    Assert(Directory.GetFiles(meetingDirectory, "*.tmp.mp3").Length == 0,
        "compression left a temporary MP3 behind");

    var previousMock = Environment.GetEnvironmentVariable("ADSUM_LOCAL_SPEECH_MOCK_INFERENCE");
    try
    {
        Environment.SetEnvironmentVariable("ADSUM_LOCAL_SPEECH_MOCK_INFERENCE", "1");
        using var localService = new MossTranscriptionService(allowExternalJobFallback: false);
        var mockTranscript = localService
            .TranscribeAsync(compressedPath, generalTerms: ["CERTANIA"])
            .GetAwaiter()
            .GetResult();
        Assert(mockTranscript.Contains("Speaker A", StringComparison.Ordinal),
            "the local pipeline could not transcribe from the compressed MP3 source");
    }
    finally
    {
        Environment.SetEnvironmentVariable("ADSUM_LOCAL_SPEECH_MOCK_INFERENCE", previousMock);
    }

    var metrics = MeetingRecorder.MeasureWaveFile(originalPath);
    var source = new RecordingResult(
        "Compression and versions",
        meetingDirectory,
        new DateTime(2026, 8, 21, 9, 0, 0),
        metrics.Duration,
        null,
        null,
        originalPath,
        null,
        null,
        new TrackMetrics(null, TimeSpan.Zero, 0, 0),
        new TrackMetrics(null, TimeSpan.Zero, 0, 0),
        metrics);
    var local = TranscriptionModelCatalog.Resolve(TranscriptionModelCatalog.LocalWhisperPyannoteId);
    var cloud = TranscriptionModelCatalog.Resolve(TranscriptionModelCatalog.GptTranscribeId);
    var localSaved = MeetingArtifactStore.SaveTranscript(
        source,
        "Speaker A: local text",
        local,
        compressedPath,
        ["CERTANIA"],
        generatedTopic: null);
    var localPath = localSaved.TranscriptPath!;
    var cloudSaved = MeetingArtifactStore.SaveTranscript(
        localSaved,
        "Cloud text",
        cloud,
        compressedPath,
        ["CERTANIA"],
        generatedTopic: null);
    Assert(File.Exists(localPath), "a second model removed the first model transcript");
    Assert(File.Exists(cloudSaved.TranscriptPath), "the second model transcript was not saved");
    Assert(!string.Equals(localPath, cloudSaved.TranscriptPath, StringComparison.OrdinalIgnoreCase),
        "different models wrote to the same transcript path");
    File.WriteAllText(Path.Combine(meetingDirectory, "transcription-compression-and-versions.md"), "legacy");

    var meeting = new MeetingLibraryService(libraryRoot).GetMeetings().Single();
    Assert(meeting.HasCompressedRecording, "library did not recognize the compressed MP3");
    Assert(meeting.TranscriptVersions.Count == 3, "library did not retain all transcript versions");
    Assert(meeting.TranscriptVersions.Any(version => version.ModelId == TranscriptionModelCatalog.LocalWhisperPyannoteId),
        "library did not identify the local model transcript");
    Assert(meeting.TranscriptVersions.Any(version => version.ModelId == TranscriptionModelCatalog.GptTranscribeId),
        "library did not identify the GPT Transcribe transcript");
    Assert(meeting.TranscriptVersions.Any(version => version.ModelId == TranscriptionModelCatalog.LegacyId),
        "library did not preserve the legacy transcript");
    Assert(File.ReadAllText(cloudSaved.TranscriptPath!).Contains("General terms applied: CERTANIA"),
        "transcript metadata did not record applied general terms");

    Console.WriteLine($"PASS compression_bytes={firstLength} local_mp3_source=true transcript_versions={meeting.TranscriptVersions.Count}");
}

void RunDiarizedAlignmentChecks()
{
    Assert(
        TranscriptionModelCatalog.All.All(model => model.IncludesSpeakerDiarization),
        "the model catalog still contains a choice without speaker diarization");

    var speakerSegments = new[]
    {
        new DiarizedTextSegment(
            TimeSpan.Zero,
            TimeSpan.FromSeconds(4),
            "speaker_0",
            "Welcome to Add some and Sir Tania."),
        new DiarizedTextSegment(
            TimeSpan.FromSeconds(4),
            TimeSpan.FromSeconds(8),
            "speaker_1",
            "We use Luca net for finance.")
    };
    var corrected = DiarizedTranscriptAligner.Align(
        "Welcome to ADsum and CERTANIA. We use LucaNet for finance.",
        speakerSegments);
    Assert(corrected.UsedAccurateText, "the normal two-pass transcript did not use GPT Transcribe wording");
    Assert(corrected.Segments.Count == 2, "alignment changed the number of speaker segments");
    Assert(corrected.Segments[0].Speaker == "speaker_0" && corrected.Segments[1].Speaker == "speaker_1",
        "alignment changed speaker identities");
    Assert(corrected.Segments[0].Text.Contains("ADsum", StringComparison.Ordinal) &&
           corrected.Segments[0].Text.Contains("CERTANIA", StringComparison.Ordinal),
        "term corrections did not remain with the first speaker");
    Assert(corrected.Segments[1].Text.Contains("LucaNet", StringComparison.Ordinal),
        "term correction did not remain with the second speaker");
    Assert(corrected.Segments[0].Start == TimeSpan.Zero &&
           corrected.Segments[1].End == TimeSpan.FromSeconds(8),
        "alignment changed speaker timestamps");

    var changedWordCount = DiarizedTranscriptAligner.Align(
        "Today we carefully review CERTANIA finance dashboard. Tomorrow publish the LucaNet report.",
        new[]
        {
            new DiarizedTextSegment(
                TimeSpan.Zero,
                TimeSpan.FromSeconds(5),
                "speaker_0",
                "Today we review the finance dashboard."),
            new DiarizedTextSegment(
                TimeSpan.FromSeconds(5),
                TimeSpan.FromSeconds(10),
                "speaker_1",
                "Tomorrow we publish the report.")
        });
    Assert(changedWordCount.UsedAccurateText, "insertions and deletions made a safe alignment fail");
    Assert(changedWordCount.Segments[0].Text.Contains("carefully", StringComparison.Ordinal),
        "an inserted first-speaker word was lost");
    Assert(changedWordCount.Segments[1].Text.Contains("LucaNet", StringComparison.Ordinal),
        "an inserted second-speaker term crossed the speaker boundary");

    var unsafeAlignment = DiarizedTranscriptAligner.Align(
        "one two three four",
        new[]
        {
            new DiarizedTextSegment(
                TimeSpan.Zero,
                TimeSpan.FromSeconds(2),
                "speaker_0",
                "alpha beta gamma delta")
        });
    Assert(unsafeAlignment.UsedAccurateText, "the fallback stopped treating GPT Transcribe as authoritative");
    Assert(unsafeAlignment.UsedProportionalFallback,
        "unrelated transcripts did not activate the proportional speaker fallback");
    Assert(unsafeAlignment.Segments[0].Text == "one two three four",
        "the fallback used diarization-model wording instead of GPT Transcribe wording");

    Console.WriteLine(
        $"PASS all_models_diarized=true alignment_ratio={corrected.ExactMatchRatio:F2} " +
        $"gpt_wording_fallback={unsafeAlignment.UsedProportionalFallback}");
}

void RunCompatibilityChecks(string directory)
{
    var microphonePath = Path.Combine(directory, "microphone.wav");
    var systemPath = Path.Combine(directory, "system.wav");
    var floatDevicePath = Path.Combine(directory, "float-device.wav");
    var emptyPath = Path.Combine(directory, "empty.wav");
    var expectedPath = Path.Combine(directory, "expected.wav");
    var actualPath = Path.Combine(directory, "actual.wav");
    var staleFloatMixPath = Path.Combine(directory, ".actual.wav.crashed.float-mix.tmp");
    var stalePendingWavePath = Path.Combine(directory, ".actual.wav.crashed.pending-wave.tmp");

    WritePcm16Fixture(microphonePath, 44100, 2, 2.37, (frame, channel, rate) =>
    {
        var seconds = (double)frame / rate;
        if (seconds < 0.19 || seconds > 2.21)
        {
            return 0;
        }

        var voice = 0.13 * Math.Sin(2 * Math.PI * 173 * seconds)
            + 0.035 * Math.Sin(2 * Math.PI * 347 * seconds);
        return (float)(voice * (channel == 0 ? 1.0 : 0.72));
    });

    WritePcm16Fixture(systemPath, 48000, 1, 3.11, (frame, _, rate) =>
    {
        var seconds = (double)frame / rate;
        if ((seconds > 0.65 && seconds < 1.34) || (seconds > 1.79 && seconds < 2.92))
        {
            return (float)(0.031 * Math.Sin(2 * Math.PI * 251 * seconds));
        }

        return 0;
    });

    WriteFloatFixture(floatDevicePath, 32000, 2, 1.73, (frame, channel, rate) =>
    {
        var seconds = (double)frame / rate;
        var envelope = seconds is > 0.11 and < 1.59 ? 1.0 : 0.0;
        return (float)(envelope * (channel == 0 ? 0.22 : -0.04) * Math.Sin(2 * Math.PI * 421 * seconds));
    });
    using (new WaveFileWriter(emptyPath, new WaveFormat(16000, 16, 1)))
    {
    }

    var inputs = new[]
    {
        microphonePath,
        Path.Combine(directory, "missing.wav"),
        systemPath,
        emptyPath,
        floatDevicePath
    };
    LegacyMixWaveFiles(inputs, expectedPath);
    File.WriteAllBytes(staleFloatMixPath, new byte[] { 1, 2, 3 });
    File.WriteAllBytes(stalePendingWavePath, new byte[] { 4, 5, 6 });
    MeetingRecorder.MixWaveFiles(inputs, actualPath);

    var expectedBytes = File.ReadAllBytes(expectedPath);
    var actualBytes = File.ReadAllBytes(actualPath);
    Assert(expectedBytes.AsSpan().SequenceEqual(actualBytes),
        $"streaming mix differs from the legacy output ({expectedBytes.Length} versus {actualBytes.Length} bytes)");

    var expectedMetrics = LegacyMeasureWaveFile(microphonePath);
    var actualMetrics = MeetingRecorder.MeasureWaveFile(microphonePath);
    Assert(expectedMetrics.Duration == actualMetrics.Duration, "measurement duration changed");
    Assert(expectedMetrics.Peak == actualMetrics.Peak, "measurement peak changed");
    Assert(expectedMetrics.Rms == actualMetrics.Rms, "measurement RMS changed");
    Assert(!File.Exists(staleFloatMixPath), "crashed float-mix temporary file was not cleaned up");
    Assert(!File.Exists(stalePendingWavePath), "crashed pending-WAV temporary file was not cleaned up");

    var temporaryFiles = Directory.GetFiles(directory, "*.tmp", SearchOption.TopDirectoryOnly);
    Assert(temporaryFiles.Length == 0, "successful mixing left temporary files behind");

    Console.WriteLine($"PASS exact_mix_bytes={actualBytes.Length} duration={actualMetrics.Duration.TotalSeconds:F6}s peak={actualMetrics.Peak:R} rms={actualMetrics.Rms:R}");
}

void RunMemoryCheck(string directory, int minutes)
{
    Assert(minutes > 0, "memory-check duration must be positive");
    var inputPath = Path.Combine(directory, "long-silent.wav");
    var outputPath = Path.Combine(directory, "long-mixed.wav");
    WriteSparseSilentPcm16Wave(inputPath, mixedSampleRate, TimeSpan.FromMinutes(minutes));

    var process = Process.GetCurrentProcess();
    process.Refresh();
    var baseline = process.PrivateMemorySize64;
    long observedPeak = baseline;
    using var stop = new CancellationTokenSource();
    var sampler = Task.Run(async () =>
    {
        while (!stop.IsCancellationRequested)
        {
            process.Refresh();
            observedPeak = Math.Max(observedPeak, process.PrivateMemorySize64);
            await Task.Delay(5);
        }
    });

    var stopwatch = Stopwatch.StartNew();
    MeetingRecorder.MixWaveFiles(new[] { inputPath }, outputPath);
    stopwatch.Stop();
    stop.Cancel();
    sampler.GetAwaiter().GetResult();
    process.Refresh();
    observedPeak = Math.Max(observedPeak, process.PrivateMemorySize64);

    var metrics = MeetingRecorder.MeasureWaveFile(outputPath);
    var expectedDuration = TimeSpan.FromMinutes(minutes);
    Assert(Math.Abs((metrics.Duration - expectedDuration).TotalMilliseconds) < 1, "long output duration changed");
    Assert(metrics.Peak == 0 && metrics.Rms == 0, "silent fixture did not remain silent");

    Console.WriteLine(
        $"PASS minutes={minutes} elapsed_seconds={stopwatch.Elapsed.TotalSeconds:F3} " +
        $"baseline_mib={baseline / 1048576.0:F2} peak_mib={observedPeak / 1048576.0:F2} " +
        $"delta_mib={(observedPeak - baseline) / 1048576.0:F2}");
}

void WritePcm16Fixture(
    string path,
    int sampleRate,
    int channels,
    double durationSeconds,
    Func<int, int, int, float> sampleFactory)
{
    var frames = (int)Math.Round(durationSeconds * sampleRate);
    var buffer = new byte[4096 * channels * sizeof(short)];
    using var writer = new WaveFileWriter(path, new WaveFormat(sampleRate, 16, channels));

    for (var firstFrame = 0; firstFrame < frames; firstFrame += 4096)
    {
        var framesThisPass = Math.Min(4096, frames - firstFrame);
        for (var frameOffset = 0; frameOffset < framesThisPass; frameOffset++)
        {
            for (var channel = 0; channel < channels; channel++)
            {
                var sample = sampleFactory(firstFrame + frameOffset, channel, sampleRate);
                var pcm = (short)Math.Clamp(sample * 32767, short.MinValue, short.MaxValue);
                BitConverter.GetBytes(pcm).CopyTo(buffer, ((frameOffset * channels) + channel) * sizeof(short));
            }
        }

        writer.Write(buffer, 0, framesThisPass * channels * sizeof(short));
    }
}

void WriteFloatFixture(
    string path,
    int sampleRate,
    int channels,
    double durationSeconds,
    Func<int, int, int, float> sampleFactory)
{
    var frames = (int)Math.Round(durationSeconds * sampleRate);
    var buffer = new float[4096 * channels];
    using var writer = new WaveFileWriter(path, WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels));

    for (var firstFrame = 0; firstFrame < frames; firstFrame += 4096)
    {
        var framesThisPass = Math.Min(4096, frames - firstFrame);
        for (var frameOffset = 0; frameOffset < framesThisPass; frameOffset++)
        {
            for (var channel = 0; channel < channels; channel++)
            {
                buffer[(frameOffset * channels) + channel] =
                    sampleFactory(firstFrame + frameOffset, channel, sampleRate);
            }
        }

        writer.WriteSamples(buffer, 0, framesThisPass * channels);
    }
}

void WriteSparseSilentPcm16Wave(string path, int sampleRate, TimeSpan duration)
{
    var sampleCount = checked((long)Math.Round(duration.TotalSeconds * sampleRate));
    var dataLength = checked(sampleCount * sizeof(short));
    Assert(dataLength <= uint.MaxValue - 36, "fixture is too large for a standard RIFF WAV");

    using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
    using var writer = new BinaryWriter(stream);
    writer.Write("RIFF"u8);
    writer.Write((uint)(36 + dataLength));
    writer.Write("WAVE"u8);
    writer.Write("fmt "u8);
    writer.Write(16u);
    writer.Write((ushort)1);
    writer.Write((ushort)1);
    writer.Write((uint)sampleRate);
    writer.Write((uint)(sampleRate * sizeof(short)));
    writer.Write((ushort)sizeof(short));
    writer.Write((ushort)16);
    writer.Write("data"u8);
    writer.Write((uint)dataLength);
    stream.SetLength(44 + dataLength);
}

void LegacyMixWaveFiles(IReadOnlyList<string> paths, string outputPath)
{
    var tracks = paths
        .Where(File.Exists)
        .Select(LegacyReadMonoSamples)
        .Where(track => track.Samples.Length > 0)
        .Select(track => LegacyResample(track.Samples, track.SampleRate, mixedSampleRate))
        .Select(LegacyNormalizeForSpeechMix)
        .ToList();

    Assert(tracks.Count > 0, "legacy fixture unexpectedly has no tracks");
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

    LegacyWritePcm16(outputPath, mixed, mixedSampleRate);
}

TrackMetrics LegacyMeasureWaveFile(string path)
{
    var track = LegacyReadMonoSamples(path);
    var peak = track.Samples.Select(Math.Abs).DefaultIfEmpty(0).Max();
    var rms = (float)Math.Sqrt(track.Samples.Select(sample => sample * sample).DefaultIfEmpty(0).Average());
    return new TrackMetrics(path, TimeSpan.FromSeconds((double)track.Samples.Length / track.SampleRate), peak, rms);
}

(float[] Samples, int SampleRate) LegacyReadMonoSamples(string path)
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

    return (samples.ToArray(), sampleRate);
}

float[] LegacyResample(float[] samples, int sourceRate, int targetRate)
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

float[] LegacyNormalizeForSpeechMix(float[] samples)
{
    var peak = samples.Select(Math.Abs).DefaultIfEmpty(0).Max();
    if (peak < minimumPeakForGain)
    {
        return samples;
    }

    var activeThreshold = Math.Max(activeThresholdFloor, peak * activePeakFraction);
    double activeSum = 0;
    var activeSamples = 0;
    foreach (var sample in samples)
    {
        if (Math.Abs(sample) < activeThreshold)
        {
            continue;
        }
        activeSum += sample * sample;
        activeSamples++;
    }

    if (activeSamples == 0)
    {
        return samples;
    }

    var activeRms = (float)Math.Sqrt(activeSum / activeSamples);
    if (activeRms < minimumActiveRms)
    {
        return samples;
    }

    var gain = Math.Clamp(targetActiveRms / activeRms, minimumTrackGain, maximumTrackGain);
    gain = Math.Min(gain, maximumNormalizedPeak / peak);
    if (Math.Abs(gain - 1.0f) < 0.01f)
    {
        return samples;
    }

    var normalized = new float[samples.Length];
    for (var index = 0; index < samples.Length; index++)
    {
        normalized[index] = samples[index] * gain;
    }
    return normalized;
}

void LegacyWritePcm16(string outputPath, float[] samples, int sampleRate)
{
    using var writer = new WaveFileWriter(outputPath, new WaveFormat(sampleRate, 16, 1));
    var bytes = new byte[samples.Length * sizeof(short)];
    for (var index = 0; index < samples.Length; index++)
    {
        var value = (short)Math.Clamp(samples[index] * 32767, short.MinValue, short.MaxValue);
        BitConverter.GetBytes(value).CopyTo(bytes, index * sizeof(short));
    }
    writer.Write(bytes, 0, bytes.Length);
}

void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}
