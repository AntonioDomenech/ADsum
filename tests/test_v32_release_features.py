from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "src" / "ADsum.Desktop" / "Services"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_every_model_choice_includes_speaker_diarization() -> None:
    source = _read(SERVICES / "TranscriptionModelOption.cs")

    assert 'LocalWhisperPyannoteId = "local-whisper-pyannote"' in source
    assert 'Gpt4oTranscribeDiarizeId = "gpt-4o-transcribe-diarize"' in source
    assert 'GptTranscribeId = "gpt-transcribe"' in source
    assert "OpenAI GPT Transcribe + Diarization" in source
    assert "IncludesSpeakerDiarization: false" not in source


def test_gpt_transcribe_runs_a_second_speaker_pass_and_aligns_the_results() -> None:
    source = _read(SERVICES / "OpenAiTranscriptionService.cs")
    aligner = _read(SERVICES / "DiarizedTranscriptAligner.cs")

    assert "Task.WhenAll(wordingTask, speakerTask)" in source
    assert "Gpt4oTranscribeDiarizeId" in source
    assert "DiarizedTranscriptAligner.Align" in source
    assert "ADsum did not save an un-diarized result" in source
    assert "GPT Transcribe wording was distributed proportionally" in aligner
    assert 'Authoritative wording: GPT Transcribe' in _read(SERVICES / "MeetingArtifactStore.cs")


def test_router_always_compresses_before_selecting_a_transcription_model() -> None:
    source = _read(SERVICES / "TranscriptionRouter.cs")

    compression = source.index("EnsureCompressedAsync(originalAudioPath")
    local = source.index("case TranscriptionModelCatalog.LocalWhisperPyannoteId")
    cloud = source.index("case TranscriptionModelCatalog.Gpt4oTranscribeDiarizeId")

    assert compression < local < cloud
    assert "compressedPath" in source


def test_compression_preserves_original_and_uses_stable_sidecar() -> None:
    source = _read(SERVICES / "AudioCompressionService.cs")

    assert 'CompressedFileName = "recording-compressed.mp3"' in source
    assert "File.Move(temporaryPath, targetPath, overwrite: true)" in source
    assert "File.Delete(source" not in source
    assert "ValidateMp3(temporaryPath, ReadDuration(fullSourcePath))" in source


def test_transcripts_are_model_specific_and_include_provenance() -> None:
    source = _read(SERVICES / "MeetingArtifactStore.cs")

    assert "TranscriptFileNameForModel" in source
    assert 'builder.AppendLine($"Transcription model:' in source
    assert 'builder.AppendLine($"Speaker diarization:' in source
    assert 'builder.AppendLine($"Transcription audio:' in source


def test_general_settings_and_transcript_version_controls_are_visible() -> None:
    xaml = _read(ROOT / "src" / "ADsum.Desktop" / "MainWindow.xaml")
    settings = _read(SERVICES / "SettingsStore.cs")

    assert '<TabItem Header="General">' in xaml
    assert 'x:Name="GeneralTermsBox"' in xaml
    assert 'x:Name="RecordModelCombo"' in xaml
    assert 'x:Name="LibraryModelCombo"' in xaml
    assert 'x:Name="LibraryTranscriptVersionCombo"' in xaml
    assert "GeneralTerms" in settings
    assert "TranscriptionModelId" in settings


def test_cli_supports_reusable_compression_and_explicit_models() -> None:
    source = _read(ROOT / "src" / "ADsum.Desktop" / "App.xaml.cs")

    assert 'HasArgument(e.Args, "--compress-recordings")' in source
    assert 'ArgValue(args, "--model")' in source
    assert "TranscriptionRouter" in source


def test_completed_library_migration_cannot_be_overwritten_by_queued_progress() -> None:
    source = _read(ROOT / "src" / "ADsum.Desktop" / "MainWindow.xaml.cs")

    assert "var acceptingProgress = 1;" in source
    assert "Volatile.Read(ref acceptingProgress) == 1" in source
    assert "Volatile.Write(ref acceptingProgress, 0);" in source
    assert source.index("Volatile.Write(ref acceptingProgress, 0);") < source.index(
        'CompressionStateText.Text = result.Failed == 0'
    )
