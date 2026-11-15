# Design

## Context

See `proposal.md` for motivation and `specs/` for behavioral requirements. Key observed environment constraints:
- Model directories under `/nfs/DGX2/Workspace/avi/fb_nchc/{asr,tts}/<lang>/models` use two layouts: flat (`asr/tsu/models/config.json`) and single-subdirectory (`asr/pyu/models/pyu_asr_20261115/`, `asr/trv/models/whisper-large-v3-turbo-trv/`, `tts/tsu/models/tsu123/`).
- `pyu` ASR `generation_config.json` defines `<|formosan|>: 51866` in `lang_to_id`, whereas `tsu` and `trv` do not.
- CosyVoice3 (`CausalConditionalCFM`) initializes its flow noise once in `__init__` (`set_all_random_seed(0)`), while LLM token sampling uses the active RNG state.
- `sarcvoice` currently has `numpy 2.2.6` (incompatible with `onnxruntime-gpu 1.18.0`), while `whisper_ft` has `transformers 5.14.1` and `jiwer`.

## Goals / Non-Goals

**Goals:**
- Provide a clean, modular Python package (`src/filtering/`) and CLI (`scripts/run_filter.py`) for closed-loop CosyVoice3 TTS + Whisper ASR CER filtering.
- Support both single-utterance and batch `.jsonl` workflows without writing intermediate `.wav` files to disk for failed attempts.
- Keep TTS and ASR engines behind clean interfaces so unit tests can mock them without GPU or model weights.

**Non-Goals:**
- Training or fine-tuning Whisper ASR or CosyVoice3 TTS models.
- Distributed multi-node scheduling (single-GPU execution per process; multi-GPU scaling can be done by sharding `.jsonl` inputs across `CUDA_VISIBLE_DEVICES`).

## Decisions

### 1. Unified single-process GPU execution (with `numpy<2` in `sarcvoice` or a dedicated env)
- **Rationale**: Running CosyVoice3 and Whisper in the same Python process allows passing the synthesized waveform tensor directly in memory (`torchaudio.functional.resample` to 16 kHz) with zero disk I/O or IPC serialization overhead.
- **Alternative considered**: Persistent subprocess worker running Whisper in `whisper_ft` while CosyVoice runs in `sarcvoice`. This avoids modifying `sarcvoice` (`numpy<2` + `jiwer`), at the cost of IPC complexity and duplicate PyTorch CUDA context memory. We keep `TTSEngine` and `ASREngine` decoupled so a subprocess-backed `ASREngine` can be swapped in if needed.

### 2. Automatic flat + nested `models/` resolution
- **Rationale**: `model_resolver.py` checks for marker files directly in `models/` (`config.json` for Whisper; `cosyvoice3.yaml` or `llm.pt` for CosyVoice). If absent, it scans immediate subdirectories for those markers, auto-selecting when one valid subdirectory exists or matching `model_name` when specified.
- **Alternative considered**: Hardcoding per-language paths. Rejected because new languages (`pyu`, `trv` TTS) or updated checkpoints would require code changes.

### 3. Per-attempt seeding via `set_all_random_seed(base_seed + attempt)`
- **Rationale**: Calling CosyVoice's `set_all_random_seed` before each `inference_zero_shot` / `inference_instruct2` call varies the LLM token sampling deterministically across attempts while keeping results reproducible given `base_seed`.

### 4. Whisper generation & Post-TN CER normalization
- **Rationale**: `ASREngine` inspects `generation_config.json`; if `<|formosan|>` is present in `lang_to_id`, it passes `language="<|formosan|>"` (or uses the model's default generation config otherwise). Before calling `jiwer.cer`, optional Post-TN normalization lowercases text, strips standard punctuation while preserving Formosan glottal stop characters (`ʼ`, `'`), and collapses whitespace—matching `eval_details_test_jsonl.csv` in `asr/tsu/`.

### 5. Configurable `keep_best_on_fail` fallback (default `True`)
- **Rationale**: Retaining the lowest-CER candidate marked `"passed": false` prevents losing usable borderline utterances during batch generation and lets downstream consumers filter by `passed` or `cer`. Setting `keep_best_on_fail=False` discards failed utterances cleanly.

## Risks / Trade-offs

- **VRAM usage (CosyVoice3 + Whisper-large-v3-turbo)**: Both models together require ~10–14 GB VRAM in fp16/fp32. → RTX 4090 GPUs on this machine have 24 GB VRAM each, which fits comfortably; `ASREngine` loads Whisper in `torch.float16` on CUDA.
- **`sarcvoice` NumPy version mismatch**: `numpy 2.2.6` breaks `onnxruntime-gpu 1.18.0`. → Resolve during setup by pinning `numpy<2` in the runtime conda environment (or creating a dedicated `filtering` env) and verifying imports before GPU integration testing.
- **Whisper `<|formosan|>` token inconsistency across languages**: `pyu` defines `<|formosan|>` while `tsu`/`trv` do not. → Inspect `generation_config.json` dynamically at load time rather than hardcoding `language="<|formosan|>"`.
