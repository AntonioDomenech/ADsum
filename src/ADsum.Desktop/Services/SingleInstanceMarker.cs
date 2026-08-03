namespace ADsum.Desktop.Services;

/// <summary>
/// Keeps recording and local MOSS work inside one ADsum v3 process.
///
/// The named mutex is deliberately used only as a kernel-object existence
/// marker. It is never acquired, so there is no thread-affine ownership to
/// release after asynchronous WPF startup work.
/// </summary>
internal sealed class SingleInstanceMarker : IDisposable
{
    private const string InstanceName =
        "ADsum.Desktop.RecordingAndMoss.8C39D10B-6E4B-4DF8-95CB-4CA658905F5D";

    private Mutex? _marker;

    private SingleInstanceMarker(Mutex marker)
    {
        _marker = marker;
    }

    public static SingleInstanceMarker? TryCreate()
    {
        var options = new NamedWaitHandleOptions
        {
            CurrentUserOnly = true,
            CurrentSessionOnly = true
        };
        var marker = new Mutex(
            initiallyOwned: false,
            InstanceName,
            options,
            out var createdNew);

        if (!createdNew)
        {
            marker.Dispose();
            return null;
        }

        return new SingleInstanceMarker(marker);
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _marker, null)?.Dispose();
    }
}
