# Spec Delta: Model Loading

## Purpose

Resolves and loads language-specific Whisper ASR and CosyVoice3 TTS models from standardized per-language model directories, supporting both flat and nested directory layouts.

## ADDED Requirements

### Requirement: Resolve flat model directory layout
The system SHALL resolve a model directory directly when model artifact files exist at the root of `/nfs/DGX2/Workspace/avi/fb_nchc/<modality>/<lang>/models`.

#### Scenario: Flat Whisper ASR directory
- **WHEN** the user selects language `tsu` and `/nfs/DGX2/Workspace/avi/fb_nchc/asr/tsu/models` directly contains `config.json`
- **THEN** the resolver returns `/nfs/DGX2/Workspace/avi/fb_nchc/asr/tsu/models` as the ASR model path

### Requirement: Resolve nested model subdirectory layout
The system SHALL resolve a unique valid model subdirectory under `/nfs/DGX2/Workspace/avi/fb_nchc/<modality>/<lang>/models/<subdir>` when model files are nested one level deep.

#### Scenario: Single nested ASR or TTS subdirectory
- **WHEN** `models/` contains a single valid model subdirectory (such as `pyu_asr_20261115` for ASR or `tsu123` for TTS)
- **THEN** the resolver returns that subdirectory path automatically

### Requirement: Select explicit model name when specified
The system SHALL select the exact subdirectory matching a caller-supplied model name when multiple subdirectories exist or an explicit override is provided.

#### Scenario: Explicit model name override
- **WHEN** the caller specifies `model_name` and `models/<model_name>` is a valid model directory
- **THEN** the resolver returns `models/<model_name>`

#### Scenario: Missing explicit model name
- **WHEN** the caller specifies a `model_name` that does not exist under `models/`
- **THEN** the resolver raises a descriptive error listing available candidates

### Requirement: Fail fast on empty or ambiguous model directories
The system SHALL raise a descriptive error if the target `models/` directory is missing, empty, or contains multiple valid model subdirectories without an explicit `model_name`.

#### Scenario: Empty models directory
- **WHEN** `models/` exists for a language (such as `tts/pyu/models`) but contains no valid model files or subdirectories
- **THEN** the resolver raises an error stating that no valid model was found in that path

#### Scenario: Multiple subdirectories without selection
- **WHEN** `models/` contains two or more valid model subdirectories and no `model_name` is specified
- **THEN** the resolver raises an error asking the caller to specify `model_name`

### Requirement: Load Whisper ASR and CosyVoice3 TTS models
The system SHALL load the resolved Whisper ASR model and CosyVoice3 TTS model onto the configured compute device for inference.

#### Scenario: Successful model initialization
- **WHEN** valid ASR and TTS model paths are resolved for language `tsu`
- **THEN** both the Whisper ASR model and CosyVoice3 TTS model are loaded and ready for inference
