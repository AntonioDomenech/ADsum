namespace ADsum.Desktop.Services;

/// <summary>
/// Gives audio recording priority over local MOSS inference.
/// Only one MOSS lease can exist in this process, and beginning a recording
/// cancels the active lease and waits for the worker to release it.
/// </summary>
public sealed class RecordingMossResourceCoordinator
{
    private readonly object _sync = new();
    private readonly SemaphoreSlim _mossGate = new(1, 1);
    private bool _recordingActive;
    private TaskCompletionSource _recordingEnded = CompletedSource();
    private TaskCompletionSource _mossIdle = CompletedSource();
    private CancellationTokenSource? _activeMossPreemption;

    private RecordingMossResourceCoordinator()
    {
    }

    public static RecordingMossResourceCoordinator Shared { get; } = new();

    public bool IsRecordingActive
    {
        get
        {
            lock (_sync)
            {
                return _recordingActive;
            }
        }
    }

    /// <summary>
    /// Marks recording as active, preempts local inference, and does not return
    /// until the inference lease has been released and its worker has exited.
    /// </summary>
    public async Task BeginRecordingAsync()
    {
        Task mossIdle;
        lock (_sync)
        {
            if (!_recordingActive)
            {
                _recordingActive = true;
                _recordingEnded = PendingSource();
            }

            _activeMossPreemption?.Cancel();
            mossIdle = _mossIdle.Task;
        }

        await mossIdle.ConfigureAwait(true);
    }

    public void EndRecording()
    {
        TaskCompletionSource? recordingEnded = null;
        lock (_sync)
        {
            if (!_recordingActive)
            {
                return;
            }

            _recordingActive = false;
            recordingEnded = _recordingEnded;
        }

        recordingEnded.TrySetResult();
    }

    public async Task WaitForRecordingToEndAsync(CancellationToken cancellationToken)
    {
        Task waitTask;
        lock (_sync)
        {
            waitTask = _recordingActive ? _recordingEnded.Task : Task.CompletedTask;
        }

        await waitTask.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    public async Task<MossLease> AcquireMossLeaseAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            await WaitForRecordingToEndAsync(cancellationToken).ConfigureAwait(false);
            await _mossGate.WaitAsync(cancellationToken).ConfigureAwait(false);

            Task? recordingEnded = null;
            lock (_sync)
            {
                if (_recordingActive)
                {
                    recordingEnded = _recordingEnded.Task;
                }
                else
                {
                    var preemption = new CancellationTokenSource();
                    var linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource(
                        cancellationToken,
                        preemption.Token);
                    _activeMossPreemption = preemption;
                    _mossIdle = PendingSource();
                    return new MossLease(this, preemption, linkedCancellation);
                }
            }

            _mossGate.Release();
            await recordingEnded!.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    private void ReleaseMossLease(
        CancellationTokenSource preemption,
        CancellationTokenSource linkedCancellation)
    {
        TaskCompletionSource? mossIdle = null;
        lock (_sync)
        {
            if (ReferenceEquals(_activeMossPreemption, preemption))
            {
                _activeMossPreemption = null;
                mossIdle = _mossIdle;
            }
        }

        linkedCancellation.Dispose();
        preemption.Dispose();
        _mossGate.Release();
        mossIdle?.TrySetResult();
    }

    private static TaskCompletionSource PendingSource() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static TaskCompletionSource CompletedSource()
    {
        var source = PendingSource();
        source.SetResult();
        return source;
    }

    public sealed class MossLease : IDisposable
    {
        private readonly RecordingMossResourceCoordinator _owner;
        private readonly CancellationTokenSource _preemption;
        private readonly CancellationTokenSource _linkedCancellation;
        private int _disposed;

        internal MossLease(
            RecordingMossResourceCoordinator owner,
            CancellationTokenSource preemption,
            CancellationTokenSource linkedCancellation)
        {
            _owner = owner;
            _preemption = preemption;
            _linkedCancellation = linkedCancellation;
        }

        public CancellationToken CancellationToken => _linkedCancellation.Token;

        public bool WasPreempted => _preemption.IsCancellationRequested;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _owner.ReleaseMossLease(_preemption, _linkedCancellation);
        }
    }
}
