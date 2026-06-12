using System.IO;
using System.Windows;
using System.Text.Json;
using ADsum.Desktop.Services;

namespace ADsum.Desktop;

public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        if (e.Args.Contains("--list-devices"))
        {
            await WriteDeviceListAsync(e.Args);
            return;
        }

        if (e.Args.Contains("--smoke-test"))
        {
            await RunSmokeTestAsync(e.Args);
            return;
        }

        new MainWindow().Show();
    }

    private static async Task RunSmokeTestAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-smoke-result.json");
        try
        {
            var duration = double.TryParse(ArgValue(args, "--duration"), out var value) ? value : 4.0;
            var micContains = ArgValue(args, "--mic-contains");
            var outputContains = ArgValue(args, "--output-contains");

            var devices = new AudioDeviceService();
            var microphone = PickDevice(devices.GetMicrophones(), micContains);
            var output = PickDevice(devices.GetRenderDevices(), outputContains);
            var recorder = new MeetingRecorder();
            var result = await recorder.RunDeviceTestAsync(
                "Smoke test",
                microphone.Id,
                output.Id,
                TimeSpan.FromSeconds(duration));

            var payload = new
            {
                ok = true,
                microphone = microphone.Name,
                output = output.Name,
                result
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static async Task WriteDeviceListAsync(string[] args)
    {
        var resultPath = ArgValue(args, "--result") ?? Path.Combine(Path.GetTempPath(), "adsum-devices.json");
        try
        {
            var devices = new AudioDeviceService();
            var payload = new
            {
                ok = true,
                microphones = devices.GetMicrophones(),
                outputs = devices.GetRenderDevices()
            };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(0);
        }
        catch (Exception ex)
        {
            var payload = new { ok = false, error = ex.ToString() };
            EnsureParentDirectory(resultPath);
            await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
            Current.Shutdown(1);
        }
    }

    private static AudioDeviceInfo PickDevice(IReadOnlyList<AudioDeviceInfo> devices, string? contains)
    {
        if (!string.IsNullOrWhiteSpace(contains))
        {
            var match = devices.FirstOrDefault(device => device.Name.Contains(contains, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
            {
                return match;
            }
        }

        return devices.FirstOrDefault(device => device.IsDefault)
            ?? devices.FirstOrDefault()
            ?? throw new InvalidOperationException("No matching audio device was found.");
    }

    private static string? ArgValue(string[] args, string name)
    {
        for (var index = 0; index < args.Length - 1; index++)
        {
            if (args[index].Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }
        return null;
    }

    private static void EnsureParentDirectory(string path)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }
}
