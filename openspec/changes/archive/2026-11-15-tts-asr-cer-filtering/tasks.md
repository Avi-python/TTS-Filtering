# Tasks

## 1. Project Scaffolding & Environment Setup

- [x] 1.1 Create `pyproject.toml`, `src/filtering/__init__.py`, and `src/filtering/config.py` with `FilterConfig`, `AttemptResult`, and `FilterResult` dataclasses; verify `python -c "from filtering.config import FilterConfig"` succeeds.
- [x] 1.2 Verify/prepare a Python runtime environment where `cosyvoice` (`onnxruntime-gpu` + `numpy<2`), `transformers`, `torchaudio`, and `jiwer` all import cleanly in the same process.

## 2. Model Path Resolution

- [x] 2.1 Implement `src/filtering/model_resolver.py` supporting flat `models/` layouts, single-subdirectory `models/<subdir>/` layouts, explicit `model_name` selection, and descriptive errors for missing/empty/ambiguous directories.
- [x] 2.2 Add `tests/test_model_resolver.py` covering flat, nested, explicit override, empty, and ambiguous layouts; verify `pytest tests/test_model_resolver.py -v` passes.

## 3. TTS & ASR Engines

- [x] 3.1 Implement `src/filtering/tts_engine.py` wrapping CosyVoice3 `AutoModel`, per-attempt `set_all_random_seed(seed)`, `inference_zero_shot`, and `inference_instruct2`, returning `(waveform, sample_rate)`; verify import and interface with unit test/smoke check.
- [x] 3.2 Implement `src/filtering/asr_engine.py` wrapping `WhisperProcessor` and `WhisperForConditionalGeneration`, `<|formosan|>` detection from `generation_config.json`, 16 kHz resampling, Post-TN text normalization, and `jiwer.cer` scoring; add unit tests for normalization and CER scoring in `tests/test_asr_and_pipeline.py`.

## 4. Filtering Pipeline & CLI

- [x] 4.1 Implement `src/filtering/pipeline.py` (`TTSFilterPipeline`) orchestrating the seed-retry loop, early exit when `cer <= cer_threshold`, and `keep_best_on_fail` fallback; add mocked unit tests in `tests/test_asr_and_pipeline.py` and verify `pytest tests/ -v` passes.
- [x] 4.2 Implement `scripts/run_filter.py` CLI supporting single-text mode and batch `.jsonl` mode (writing filtered `.wav` files and output `.jsonl` metadata) plus `README.md` usage examples; verify `--help` exits 0.

## 5. GPU Integration Verification

- [x] 5.1 Run `scripts/run_filter.py` on GPU with `tsu` ASR (`/nfs/DGX2/Workspace/avi/fb_nchc/asr/tsu/models`) and `tsu` TTS (`/nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/models/tsu123`) on 2–3 sample utterances and verify generated `.wav` files and `.jsonl` CER/seed/passed metadata.
