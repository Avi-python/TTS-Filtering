# Spec Delta: CER Filtering

## Purpose

Synthesizes speech with CosyVoice3 using controlled random seeds, transcribes the audio with Whisper ASR, and retries with new seeds until the Character Error Rate meets a threshold or max retries are reached.

## ADDED Requirements

### Requirement: Synthesize speech with deterministic seed per attempt
The system SHALL set the random seed before each CosyVoice3 synthesis attempt so that retries produce distinct, reproducible candidates.

#### Scenario: Seed progression across retries
- **WHEN** `base_seed` is `42` and attempt `0` fails the CER check
- **THEN** attempt `0` uses seed `42` and attempt `1` uses seed `43`

### Requirement: Evaluate synthesized audio with Whisper ASR and CER
The system SHALL resample synthesized TTS audio to 16 kHz, transcribe it with the language-matched Whisper ASR model, and compute CER against the target text.

#### Scenario: In-memory resampling and CER scoring
- **WHEN** CosyVoice3 outputs speech at its native sample rate
- **THEN** the system resamples it to 16000 Hz, runs Whisper transcription, and computes CER between the target text and ASR hypothesis

### Requirement: Optional Post-TN text normalization before CER
The system SHALL support normalizing both reference text and ASR hypothesis (lowercasing, stripping punctuation while preserving Formosan glottal characters, and collapsing whitespace) before computing CER when normalization is enabled.

#### Scenario: Normalized CER comparison
- **WHEN** `normalize_text` is enabled and the target text contains trailing punctuation or uppercase letters
- **THEN** both target text and ASR hypothesis are normalized before CER is calculated

### Requirement: Accept early when CER is within threshold
The system SHALL stop retrying as soon as an attempt achieves `CER <= cer_threshold`, save the synthesized audio, and mark the result as passed.

#### Scenario: Pass on retry attempt
- **WHEN** `cer_threshold` is `0.15`, attempt `0` yields CER `0.25`, and attempt `1` yields CER `0.10`
- **THEN** the system stops after attempt `1`, saves attempt `1`'s audio, and outputs `passed: true` with `attempts: 2`

### Requirement: Handle max retries exhaustion
The system SHALL stop after `max_retries` attempts if no attempt achieves `CER <= cer_threshold`, and apply the configured fallback policy (`keep_best_on_fail`).

#### Scenario: Keep lowest-CER candidate on failure
- **WHEN** all `max_retries` attempts exceed `cer_threshold` and `keep_best_on_fail` is `true`
- **THEN** the system saves the audio from the attempt with the lowest CER and outputs `passed: false`

#### Scenario: Discard output on failure
- **WHEN** all `max_retries` attempts exceed `cer_threshold` and `keep_best_on_fail` is `false`
- **THEN** the system does not save an output `.wav` file and outputs `passed: false` with `audio_path: null`

### Requirement: Support single-utterance and batch JSONL modes
The system SHALL provide a CLI supporting both single-text filtering and batch `.jsonl` dataset filtering with per-item prompt override or CLI default prompts.

#### Scenario: Batch JSONL filtering
- **WHEN** the user runs the CLI with `--input_jsonl` and `--output_jsonl`
- **THEN** each input record is processed through the CER filter and written to `--output_jsonl` with `tts_audio`, `asr_hypothesis`, `cer`, `seed`, `attempts`, and `passed` fields
