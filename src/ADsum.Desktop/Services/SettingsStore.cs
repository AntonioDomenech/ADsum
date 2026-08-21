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

    public string? OpenAiKey => Unprotect(_settings.OpenAiKey) ?? FindSetting("ADSUM_OPENAI_API_KEY", "OPENAI_API_KEY");

    public string NotesModel => FindSetting("ADSUM_OPENAI_NOTES_MODEL", "ADSUM_OPENAI_MINUTES_MODEL") ?? "gpt-5.5";

    public TranscriptionModelOption SelectedTranscriptionModel =>
        TranscriptionModelCatalog.Resolve(_settings.TranscriptionModelId);

    public IReadOnlyList<string> GeneralTerms => NormalizeTerms(_settings.GeneralTerms ?? []);

    public bool HasOpenAiKey => !string.IsNullOrWhiteSpace(OpenAiKey);

    public bool UseLocalTopicNaming => IsTruthy(FindSetting("ADSUM_LOCAL_TOPIC_ONLY"));

    public void SaveOpenAiKey(string key)
    {
        _settings.OpenAiKey = Protect(key);
        Save();
    }

    public void SaveGeneralSettings(string transcriptionModelId, IEnumerable<string> generalTerms)
    {
        _settings.TranscriptionModelId = TranscriptionModelCatalog.Resolve(transcriptionModelId).Id;
        _settings.GeneralTerms = NormalizeTerms(generalTerms).ToList();
        Save();
    }

    public static IReadOnlyList<string> ParseGeneralTerms(string text)
    {
        var terms = NormalizeTerms(text.Split(new[] { "\r\n", "\n", "\r" }, StringSplitOptions.None));
        if (terms.Count > 100)
        {
            throw new InvalidOperationException("Keep the general list to 100 terms or fewer.");
        }

        foreach (var term in terms)
        {
            if (term.Length > 120)
            {
                throw new InvalidOperationException($"A general term is longer than 120 characters: {term[..Math.Min(40, term.Length)]}...");
            }
            if (term.Contains('<') || term.Contains('>'))
            {
                throw new InvalidOperationException("General terms cannot contain < or > because the transcription API rejects them.");
            }
        }

        return terms;
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

    private static string? FindSetting(params string[] names)
    {
        foreach (var name in names)
        {
            var value = NonEmpty(Environment.GetEnvironmentVariable(name));
            if (value is not null)
            {
                return value;
            }
        }

        foreach (var path in DotEnvCandidates())
        {
            foreach (var name in names)
            {
                var value = ReadDotEnvValue(path, name);
                if (value is not null)
                {
                    return value;
                }
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

    private static IReadOnlyList<string> NormalizeTerms(IEnumerable<string> values) => values
        .Select(value => value.Trim())
        .Where(value => value.Length > 0)
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToList();

    private static bool IsTruthy(string? value) => value is not null &&
        (value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
         value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
         value.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
         value.Equals("on", StringComparison.OrdinalIgnoreCase));

    private sealed class AppSettings
    {
        public string? OpenAiKey { get; set; }

        public string? TranscriptionModelId { get; set; }

        public List<string>? GeneralTerms { get; set; }
    }
}
