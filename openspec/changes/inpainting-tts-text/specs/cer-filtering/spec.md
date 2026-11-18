# Spec Delta: CER Filtering

## ADDED Requirements

### Requirement: Support separate TTS synthesis text and reference text
The system SHALL allow callers to supply a separate TTS synthesis text (such as phoneme-inpainted hotfix text) while evaluating Whisper ASR CER against the ground-truth reference text, falling back to the reference text when synthesis text is omitted or empty.

#### Scenario: Inpainted synthesis text with clean reference text
- **WHEN** the caller provides reference text `"ʼa ongko ta pasuya 我的 English 不是很好."` and synthesis text `"ʼa ongko ta pasuya 我的 [IH1][NG][G][L][IH0][SH] 不是很好."`
- **THEN** the system passes the synthesis text to CosyVoice3 TTS, computes CER between the reference text and the Whisper ASR hypothesis, and records both texts in the result

#### Scenario: Omitted synthesis text fallback
- **WHEN** the caller omits `tts_text` or provides an empty string
- **THEN** the system uses the reference text for both TTS synthesis and CER evaluation

## MODIFIED Requirements

### Requirement: Support single-utterance and batch JSONL modes
The system SHALL provide a CLI supporting both single-text filtering (with optional `--tts_text`) and batch `.jsonl` dataset filtering with per-item prompt override, opt-in `--tts_text_field` synthesis text override, or CLI default prompts.

#### Scenario: Batch JSONL filtering
- **WHEN** the user runs the CLI with `--input_jsonl` and `--output_jsonl`
- **THEN** each input record is processed through the CER filter and written to `--output_jsonl` with `tts_audio`, `tts_input_text`, `asr_hypothesis`, `cer`, `seed`, `attempts`, and `passed` fields

#### Scenario: Batch JSONL with opt-in synthesis text field
- **WHEN** the user passes `--tts_text_field inpaint_text` and a record contains a non-empty `inpaint_text` value
- **THEN** the system synthesizes speech using `inpaint_text`, evaluates CER against `--text_field` (or auto-detected reference text), and records the synthesis text in `tts_input_text`
