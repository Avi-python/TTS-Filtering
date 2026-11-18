# Proposal

## Why

When CosyVoice3 mispronounces specific words (such as loanwords, code-switched terms, or tricky phonemes), users can apply phoneme/inpainting hotfixes (e.g., `[IH1][NG][G][L][IH0][SH]`) in the TTS input text while still needing Whisper ASR CER evaluation to score against the original orthography. Currently, the filtering pipeline uses a single `text` string for both TTS synthesis and ASR CER comparison, preventing inpainting hotfixes from being used in the filter loop.

## What Changes

- Decouple the TTS synthesis input text (`tts_text`) from the ASR ground-truth reference text (`text`) in `TTSFilterPipeline` and `FilterResult`, defaulting `tts_text` to `text` when omitted.
- Add `--tts_text` for single-utterance CLI mode and `--tts_text_field` (opt-in) for batch `.jsonl` CLI mode, falling back to the reference text field when the specified field is missing or empty on a record.
- Record `tts_input_text` in CLI output metadata so existing `.jsonl` fields (such as `tts_text` in `sim_10.jsonl`) are never overwritten or misread.

## Capabilities

### New Capabilities
<!-- None: extends the existing cer-filtering capability. -->

### Modified Capabilities
- `cer-filtering`: Support separate TTS synthesis input text (`tts_text`, e.g., with phoneme inpainting hotfixes) and ASR ground-truth reference text (`text`), plus opt-in `--tts_text` / `--tts_text_field` CLI options.

## Impact

- **Code**: `src/filtering/config.py`, `src/filtering/pipeline.py`, `scripts/run_filter.py`, `tests/test_asr_and_pipeline.py`, and `README.md`.
- **Compatibility**: Fully backward-compatible; omitting `tts_text` / `--tts_text_field` preserves identical behavior to the current single-text flow.
