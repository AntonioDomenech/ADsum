using NAudio.CoreAudioApi;

namespace ADsum.Desktop.Services;

public sealed class AudioDeviceService
{
    public IReadOnlyList<AudioDeviceInfo> GetMicrophones() => GetDevices(DataFlow.Capture);

    public IReadOnlyList<AudioDeviceInfo> GetRenderDevices() => GetDevices(DataFlow.Render);

    public MMDevice GetMicrophone(string id) => GetDevice(DataFlow.Capture, id);

    public MMDevice GetRenderDevice(string id) => GetDevice(DataFlow.Render, id);

    private static IReadOnlyList<AudioDeviceInfo> GetDevices(DataFlow flow)
    {
        using var enumerator = new MMDeviceEnumerator();
        var defaultId = TryGetDefaultId(enumerator, flow);
        return enumerator
            .EnumerateAudioEndPoints(flow, DeviceState.Active)
            .Select(device => new AudioDeviceInfo(
                device.ID,
                CleanName(device.FriendlyName) + (device.ID == defaultId ? " (default)" : ""),
                device.ID == defaultId,
                BluetoothWarning(device.FriendlyName, flow)))
            .ToList();
    }

    private static MMDevice GetDevice(DataFlow flow, string id)
    {
        using var enumerator = new MMDeviceEnumerator();
        if (string.IsNullOrWhiteSpace(id))
        {
            return enumerator.GetDefaultAudioEndpoint(flow, Role.Multimedia);
        }

        return enumerator.GetDevice(id);
    }

    private static string? TryGetDefaultId(MMDeviceEnumerator enumerator, DataFlow flow)
    {
        try
        {
            return enumerator.GetDefaultAudioEndpoint(flow, Role.Multimedia).ID;
        }
        catch
        {
            return null;
        }
    }

    private static string CleanName(string value) => string.Join(" ", value.Split(default(string[]), StringSplitOptions.RemoveEmptyEntries));

    private static string? BluetoothWarning(string name, DataFlow flow)
    {
        var lower = name.ToLowerInvariant();
        if (lower.Contains("headset") || lower.Contains("hands-free"))
        {
            return "Bluetooth headset microphones can switch Windows into hands-free mode. If playback quality drops, try a separate mic.";
        }
        if (flow == DataFlow.Render && (lower.Contains("buds") || lower.Contains("bluetooth")))
        {
            return "Confirm this is the output device you are listening through before recording.";
        }
        return null;
    }
}
