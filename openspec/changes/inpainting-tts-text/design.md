# Design

## Context

See `proposal.md` for motivation and `specs/cer-filtering/spec.md` for behavioral requirements. Existing datasets (e.g., `/nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/sim_10/sim_10.jsonl`) already contain a `tts_text` field used for speaker-similarity evaluation that differs from `clean_text`.

## Goals / Non-Goals

**Goals:**
- Decouple TTS synthesis text (`tts_text`) from ASR reference text (`text`) across `FilterConfig`/`FilterResult`, `TTSFilterPipeline`, and `scripts/run_filter.py`.
- Keep existing CLI/JSONL behavior 100% backward-compatible when no synthesis override is supplied.

**Non-Goals:**
- Automatic phoneme/G2P conversion or inpainting generation (callers provide the hotfix text directly).

## Decisions

### 1. Opt-in `--tts_text_field` in batch JSONL mode
- **Rationale**: Because existing `.jsonl` files like `sim_10.jsonl` already contain an unrelated `tts_text` column, auto-reading `tts_text` by default would silently synthesize the wrong sentence. Requiring explicit `--tts_text_field <name>` in batch mode eliminates accidental collisions while supporting per-record fallback to the reference text when the field is missing or empty.
- **Alternative considered**: Auto-reading a fixed new field name such as `inpaint_text`. Rejected for now to avoid implicit schema magic; can be added later if a standard field name is adopted.

### 2. Output field named `tts_input_text`
- **Rationale**: Writing `tts_input_text` instead of overwriting `tts_text` in output `.jsonl` preserves all input columns intact while recording the exact string passed to CosyVoice3.

## Risks / Trade-offs

- **Misleading CER if hotfix changes spoken content**: If a user's `tts_text` alters actual words rather than pronunciation/phonemes, CER against `text` will increase and trigger retries as expected.
