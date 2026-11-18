# Tasks

## 1. Pipeline & Result Model

- [x] 1.1 Update `FilterResult` in `src/filtering/config.py` to include `tts_text: str` (defaulting to `""` or initialized from pipeline) and update `TTSFilterPipeline.filter_utterance` in `src/filtering/pipeline.py` to accept `tts_text: Optional[str] = None`, using `tts_text or text` for TTS synthesis while scoring CER against `text`.
- [x] 1.2 Add unit tests in `tests/test_asr_and_pipeline.py` verifying separate `tts_text` synthesis vs `text` CER evaluation and fallback when `tts_text` is `None` or empty; verify `pytest tests/ -v` passes.

## 2. CLI & Documentation

- [x] 2.1 Update `scripts/run_filter.py` to add `--tts_text` (single-text mode) and `--tts_text_field` (opt-in batch JSONL mode with fallback to reference text when missing/empty), recording `tts_input_text` in JSON/JSONL outputs; verify `python scripts/run_filter.py --help` exits 0 and add CLI unit/smoke test.
- [x] 2.2 Update `README.md` with inpainting hotfix examples (`--tts_text` and `--tts_text_field`); verify examples match actual CLI flags.
