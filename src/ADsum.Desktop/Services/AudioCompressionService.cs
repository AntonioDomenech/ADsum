using System.Collections.Concurrent;
using System.IO;
using NAudio.MediaFoundation;
using NAudio.Wave;

namespace ADsum.Desktop.Services;

public sealed class AudioCompressionService
{
    public const string CompressedFileName = "recording-compressed.mp3";
    public const int SpeechBitRate = 32000;
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> FileLocks =
        new(StringComparer.OrdinalIgnoreCase);

    public string CompressedPathFor(string sourceAudioPath)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(sourceAudioPath))
            ?? throw new InvalidOperationException("Audio file has no parent directory.");
        return Path.Combine(directory, CompressedFileName);
    }

    public async Task<string> EnsureCompressedAsync(
        string sourceAudioPath,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(sourceAudioPath))
        {
            throw new FileNotFoundException("Original audio file was not found.", sourceAudioPath);
        }

        var fullSourcePath = Path.GetFullPath(sourceAudioPath);
        var targetPath = CompressedPathFor(fullSourcePath);
        var gate = FileLocks.GetOrAdd(targetPath, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (IsReusable(fullSourcePath, targetPath))
            {
                progress?.Report("Using existing compressed MP3");
                return targetPath;
            }

            progress?.Report("Compressing recording to MP3");
            var temporaryPath = Path.Combine(
                Path.GetDirectoryName(targetPath)!,
                $".{Path.GetFileNameWithoutExtension(targetPath)}.{Guid.NewGuid():N}.tmp.mp3");
            try
            {
                await Task.Run(
                    () => EncodeMp3(fullSourcePath, temporaryPath, cancellationToken),
                    cancellationToken).ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                ValidateMp3(temporaryPath, ReadDuration(fullSourcePath));
                File.Move(temporaryPath, targetPath, overwrite: true);
                File.SetLastWriteTimeUtc(targetPath, File.GetLastWriteTimeUtc(fullSourcePath));
                return targetPath;
            }
            finally
            {
                TryDelete(temporaryPath);
            }
        }
        finally
        {
            gate.Release();
        }
    }

    public async Task<AudioCompressionBatchResult> CompressLibraryAsync(
        string recordingsRoot,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(recordingsRoot))
        {
            return new AudioCompressionBatchResult(0, 0, 0, []);
        }

        var sources = Directory
            .EnumerateDirectories(recordingsRoot)
            .Select(FindOriginalRecording)
            .Where(path => path is not null)
            .Cast<string>()
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();
        var converted = 0;
        var reused = 0;
        var failures = new List<AudioCompressionFailure>();

        for (var index = 0; index < sources.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var source = sources[index];
            var target = CompressedPathFor(source);
            var alreadyReady = IsReusable(source, target);
            progress?.Report($"Compressing recording {index + 1} of {sources.Count}: {Path.GetFileName(Path.GetDirectoryName(source))}");
            try
            {
                await EnsureCompressedAsync(source, cancellationToken: cancellationToken).ConfigureAwait(false);
                if (alreadyReady)
                {
                    reused++;
                }
                else
                {
                    converted++;
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                failures.Add(new AudioCompressionFailure(source, ex.Message));
            }
        }

        return new AudioCompressionBatchResult(sources.Count, converted, reused, failures);
    }

    public static string? FindOriginalRecording(string directory)
    {
        foreach (var candidate in new[]
        {
            Path.Combine(directory, MeetingArtifactStore.RecordingFileName),
            Path.Combine(directory, "mixed.wav")
        })
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return Directory
            .EnumerateFiles(directory, "recording-*.wav")
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .FirstOrDefault();
    }

    public static bool IsReusable(string sourcePath, string compressedPath)
    {
        if (!File.Exists(sourcePath) || !File.Exists(compressedPath))
        {
            return false;
        }

        var compressed = new FileInfo(compressedPath);
        if (compressed.Length <= 0 || compressed.LastWriteTimeUtc < File.GetLastWriteTimeUtc(sourcePath))
        {
            return false;
        }

        try
        {
            ValidateMp3(compressedPath, ReadDuration(sourcePath));
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static void EncodeMp3(string sourcePath, string outputPath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var reader = new AudioFileReader(sourcePath);
        MediaFoundationApi.Startup();
        MediaFoundationEncoder.EncodeToMp3(reader, outputPath, SpeechBitRate);
        cancellationToken.ThrowIfCancellationRequested();
    }

    private static TimeSpan ReadDuration(string path)
    {
        using var reader = new AudioFileReader(path);
        return reader.TotalTime;
    }

    private static void ValidateMp3(string path, TimeSpan expectedDuration)
    {
        if (!File.Exists(path) || new FileInfo(path).Length <= 0)
        {
            throw new InvalidDataException("MP3 compression produced an empty file.");
        }

        using var reader = new Mp3FileReader(path);
        if (reader.TotalTime <= TimeSpan.Zero)
        {
            throw new InvalidDataException("MP3 compression produced an unreadable duration.");
        }

        var tolerance = TimeSpan.FromSeconds(Math.Max(1.0, expectedDuration.TotalSeconds * 0.005));
        if (expectedDuration > TimeSpan.Zero && (reader.TotalTime - expectedDuration).Duration() > tolerance)
        {
            throw new InvalidDataException(
                $"Compressed MP3 duration {reader.TotalTime} does not match original duration {expectedDuration}.");
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // A failed temporary-file cleanup should not hide the conversion error.
        }
    }
}

public sealed record AudioCompressionFailure(string SourcePath, string Error);

public sealed record AudioCompressionBatchResult(
    int Total,
    int Converted,
    int Reused,
    IReadOnlyList<AudioCompressionFailure> Failures)
{
    public int Failed => Failures.Count;
}
