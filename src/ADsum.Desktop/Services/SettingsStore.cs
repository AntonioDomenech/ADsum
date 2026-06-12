using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace ADsum.Desktop.Services;

public sealed class SettingsStore
{
    private readonly string _settingsPath;
    private AppSettings _settings;

    public SettingsStore()
    {
        var directory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ADsum");
        Directory.CreateDirectory(directory);
        _settingsPath = Path.Combine(directory, "settings.json");
        _settings = Load();
    }

    public string? OpenAiKey => Unprotect(_settings.OpenAiKey) ?? FindOpenAiKey();

    public bool HasOpenAiKey => !string.IsNullOrWhiteSpace(OpenAiKey);

    public void SaveOpenAiKey(string key)
    {
        _settings.OpenAiKey = Protect(key);
        Save();
    }

    private AppSettings Load()
    {
        if (!File.Exists(_settingsPath))
        {
            return new AppSettings();
        }

        try
        {
            return JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(_settingsPath)) ?? new AppSettings();
        }
        catch
        {
            return new AppSettings();
        }
    }

    private void Save()
    {
        File.WriteAllText(_settingsPath, JsonSerializer.Serialize(_settings, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static string Protect(string value)
    {
        var bytes = ProtectedData.Protect(Encoding.UTF8.GetBytes(value), null, DataProtectionScope.CurrentUser);
        return Convert.ToBase64String(bytes);
    }

    private static string? Unprotect(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        try
        {
            var bytes = ProtectedData.Unprotect(Convert.FromBase64String(value), null, DataProtectionScope.CurrentUser);
            return Encoding.UTF8.GetString(bytes);
        }
        catch
        {
            return null;
        }
    }

    private static string? FindOpenAiKey()
    {
        var key = NonEmpty(Environment.GetEnvironmentVariable("ADSUM_OPENAI_API_KEY"))
            ?? NonEmpty(Environment.GetEnvironmentVariable("OPENAI_API_KEY"));
        if (key is not null)
        {
            return key;
        }

        foreach (var path in DotEnvCandidates())
        {
            key = ReadDotEnvValue(path, "ADSUM_OPENAI_API_KEY") ?? ReadDotEnvValue(path, "OPENAI_API_KEY");
            if (key is not null)
            {
                return key;
            }
        }
        return null;
    }

    private static IEnumerable<string> DotEnvCandidates()
    {
        yield return Path.Combine(Directory.GetCurrentDirectory(), ".env");
        yield return Path.Combine(AppContext.BaseDirectory, ".env");
    }

    private static string? ReadDotEnvValue(string path, string name)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        foreach (var line in File.ReadLines(path))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }

            var separator = trimmed.IndexOf('=');
            if (separator <= 0)
            {
                continue;
            }

            var key = trimmed[..separator].Trim();
            if (!key.Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var value = trimmed[(separator + 1)..].Trim().Trim('"', '\'');
            return NonEmpty(value);
        }
        return null;
    }

    private static string? NonEmpty(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim();

    private sealed class AppSettings
    {
        public string? OpenAiKey { get; set; }
    }
}
