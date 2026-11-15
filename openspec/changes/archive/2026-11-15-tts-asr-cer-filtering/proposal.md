# Proposal

## Why

Generated Formosan TTS audio from CosyVoice3 can occasionally exhibit pronunciation errors, dropped words, or hallucinations due to stochastic LLM token sampling. We need an automated closed-loop filtering system that verifies each synthesized utterance with a language-matched Whisper ASR model and retries with a new random seed whenever the Character Error Rate (CER) exceeds a configured threshold.

## What Changes

- Introduce a new `filtering` Python package and CLI under `/mnt/usr1/avi/Projects/formosan/Filtering` to run closed-loop TTS generation and ASR verification.
- Add language-based model path resolution that automatically handles both flat (`models/config.json`) and nested (`models/<subdir>/`) directory layouts under `/nfs/DGX2/Workspace/avi/fb_nchc/asr/<lang>/models` and `/nfs/DGX2/Workspace/avi/fb_nchc/tts/<lang>/models`.
- Add CosyVoice3 TTS and Whisper ASR inference wrappers with per-attempt random seed control, in-memory resampling to 16 kHz, optional Post-TN text normalization, and CER computation via `jiwer`.
- Add a retry pipeline and CLI supporting single-text and batch `.jsonl` filtering modes, outputting accepted `.wav` files and per-utterance metadata (CER, seed, attempts, pass/fail status).

## Capabilities

### New Capabilities
- `model-loading`: Resolve and load per-language Whisper ASR and CosyVoice3 TTS models from flat or nested `models/` directories with clear errors for missing or ambiguous model directories.
- `cer-filtering`: Synthesize speech with controlled seeds, transcribe via Whisper ASR, evaluate CER against a threshold, and retry with new seeds until passing or exhausting `max_retries` in single or batch `.jsonl` mode.

### Modified Capabilities
<!-- None: greenfield project with no existing OpenSpec specs. -->

## Impact

- **New code**: `src/filtering/` (`config.py`, `model_resolver.py`, `tts_engine.py`, `asr_engine.py`, `pipeline.py`), `scripts/run_filter.py`, and `tests/` under `/mnt/usr1/avi/Projects/formosan/Filtering`.
- **External dependencies**: Consumes `/mnt/usr1/avi/Projects/formosan/cosyvoice` (CosyVoice3 `AutoModel` and `Matcha-TTS`), `transformers` (`WhisperProcessor`, `WhisperForConditionalGeneration`), `torchaudio`, `jiwer`, and model weights under `/nfs/DGX2/Workspace/avi/fb_nchc/{asr,tts}/<lang>/models`.
- **Environment note**: `sarcvoice` currently has `numpy 2.2.6` conflicting with `onnxruntime-gpu 1.18.0`, while `whisper_ft` supports Whisper + `jiwer`; implementation will either unify into a single working environment (`numpy<2` + `jiwer`) or use separate worker subprocesses.
