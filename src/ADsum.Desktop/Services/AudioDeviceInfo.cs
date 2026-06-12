namespace ADsum.Desktop.Services;

public sealed record AudioDeviceInfo(
    string Id,
    string Name,
    bool IsDefault,
    string? Warning);
