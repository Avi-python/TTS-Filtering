# Formosan TTS + ASR CER Filtering 系統使用說明

本系統整合 **CosyVoice3 TTS** 與 **Whisper ASR**，透過閉迴路（closed-loop）字元錯誤率（CER）檢查與隨機種子重試機制，自動過濾合成語音品質：

1. 依 `--lang` 自動解析並載入 `/nfs/DGX2/Workspace/avi/fb_nchc/tts/<lang>/models` 與 `/nfs/DGX2/Workspace/avi/fb_nchc/asr/<lang>/models` 的模型。
2. 使用 `seed = base_seed + attempt` 合成語音。
3. 在記憶體中將音訊重採樣至 `16000 Hz`，送入 Whisper ASR 辨識並計算 CER。
4. 若 `CER <= cer_threshold`，立即通過並儲存 `.wav`；否則更換下一個 seed 重新生成，直到通過或達到 `max_retries`。

---

## 1. 環境需求

請使用已設定好 `numpy<2` 與 `jiwer` 的 `sarcvoice` Conda 環境執行：

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python
```

若要在新機器或新環境安裝，可執行：
```bash
pip install -e /mnt/usr1/avi/Projects/formosan/Filtering
```

---

## 2. 模型目錄結構說明

系統支援以下兩種 `models/` 目錄結構（會自動偵測）：

- **扁平結構**：`models/` 底下直接放置模型檔案（例如 `/nfs/DGX2/Workspace/avi/fb_nchc/asr/tsu/models/config.json`）。
- **巢狀子目錄結構**：`models/<model_name>/` 底下放置模型檔案（例如 `/nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/models/tsu_tts_20261115/`、`asr/pyu/models/pyu_asr_20261115/`）。

若同一語言的 `models/` 下有多個模型子目錄，請加上 `--asr_model_name <子目錄名稱>` 或 `--tts_model_name <子目錄名稱>` 明確指定。

---

## 3. CLI 執行範例

### 範例 A：單句模式（Single-text Mode）

指定單一句子進行合成、ASR 驗證與 seed 重試：

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python /mnt/usr1/avi/Projects/formosan/Filtering/scripts/run_filter.py \
  --lang tsu \
  --text "mo pio na avʼu ta oko?" \
  --prompt_wav /nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav \
  --prompt_text "a panto pangka to kiosicumia." \
  --instruct_prefix "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>" \
  --cer_threshold 0.15 \
  --max_retries 5 \
  --base_seed 42 \
  --output_wav outputs/tsu_single.wav
```

執行後會在終端機輸出 JSON 摘要（包含 `passed`、`best_cer`、`best_seed`、`attempts`、`hypothesis` 與 `attempt_history`）。

---

### 範例 B：批次 JSONL 模式（Batch JSONL Mode）

從 `.jsonl` 讀取多筆語料（預設自動讀取 `clean_text`、`text` 或 `target_text` 欄位，也可用 `--text_field` 指定），並輸出過濾後的 `.wav` 與結果 `.jsonl`：

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python /mnt/usr1/avi/Projects/formosan/Filtering/scripts/run_filter.py \
  --lang tsu \
  --input_jsonl /nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/sim_10/sim_10.jsonl \
  --prompt_wav /nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav \
  --prompt_text "a panto pangka to kiosicumia." \
  --instruct_prefix "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>" \
  --cer_threshold 0.15 \
  --max_retries 5 \
  --output_dir outputs/tsu_batch \
  --output_jsonl outputs/tsu_batch/results.jsonl
```

> [!TIP]
> 若只想先試跑前幾筆資料，可加上 `--limit 5`。若要指定特定 GPU，可在指令前加上 `CUDA_VISIBLE_DEVICES=0`（未設定時程式會自動挑選剩餘 VRAM 最多的 GPU）。

---

### 範例 C：嚴格過濾（未通過時不保留音檔）

預設情況下，若重試 `max_retries` 次仍未通過門檻，系統會保留 CER 最低的音檔並標記 `"passed": false`。若希望未通過時**不輸出 `.wav`**，請加上 `--discard_on_fail`：

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python /mnt/usr1/avi/Projects/formosan/Filtering/scripts/run_filter.py \
  --lang tsu \
  --input_jsonl input.jsonl \
  --prompt_wav /path/to/prompt.wav \
  --prompt_text "prompt text" \
  --cer_threshold 0.10 \
  --max_retries 5 \
  --discard_on_fail \
  --output_dir outputs/tsu_strict \
  --output_jsonl outputs/tsu_strict/results.jsonl
```

---

## 4. Python API 使用範例

您也可以直接在 Python 程式中呼叫 `FilterConfig` 與 `TTSFilterPipeline`：

```python
from pathlib import Path
from filtering.config import FilterConfig
from filtering.pipeline import TTSFilterPipeline

config = FilterConfig(
    lang="tsu",
    cer_threshold=0.15,
    max_retries=5,
    base_seed=42,
    default_prompt_wav="/nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav",
    default_prompt_text="a panto pangka to kiosicumia.",
    instruct_prefix="You are a helpful assistant. 请用鄒語表达。<|endofprompt|>",
)

pipeline = TTSFilterPipeline.from_config(config)
result = pipeline.filter_utterance(
    text="mo pio na avʼu ta oko?",
    output_wav_path=Path("outputs/api_sample.wav"),
)

print(
    f"passed={result.passed}, cer={result.best_attempt.cer:.4f}, "
    f"seed={result.best_attempt.seed}, hyp={result.best_attempt.hypothesis}"
)
```

---

## 5. 常用參數一覽

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--lang` | （必填） | 語言代碼（如 `tsu`、`pyu`、`trv`） |
| `--cer_threshold` | `0.15` | 通過門檻（`CER <= cer_threshold` 即通過） |
| `--max_retries` | `5` | 單句最多嘗試的 seed 次數 |
| `--base_seed` | `42` | 起始 seed（第 `i` 次嘗試使用 `base_seed + i`） |
| `--tts_mode` | `zero_shot` | TTS 推論模式（`zero_shot` 或 `instruct2`） |
| `--prompt_wav` | `None` | 預設參考音檔路徑（若 JSONL 內含 `prompt_wav` 欄位則優先使用該筆設定） |
| `--prompt_text` | `None` | 預設參考音檔文本 |
| `--instruct_prefix` | `You are a helpful assistant.<|endofprompt|>` | 指令前綴（鄒語建議使用 `You are a helpful assistant. 请用鄒語表达。<|endofprompt|>`） |
| `--asr_model_name` | `None` | 指定 `asr/<lang>/models/` 下的特定子目錄名稱 |
| `--tts_model_name` | `None` | 指定 `tts/<lang>/models/` 下的特定子目錄名稱 |
| `--no_normalize` | `False` | 停用 Post-TN 文本正規化（預設會轉小寫並去除標點，保留 `ʼ` / `'` 等喉塞音符號） |
| `--discard_on_fail` | `False` | 超過 `max_retries` 仍未通過時捨棄 `.wav`（預設保留 CER 最低者並標記 `passed: false`） |

---

## 6. 輸出 JSONL 欄位說明

批次模式輸出的 `.jsonl` 會保留原始輸入的所有欄位，並新增以下欄位：

- `tts_audio`：最終儲存的 `.wav` 檔案路徑（若 `--discard_on_fail` 且未通過則為 `null`）
- `asr_hypothesis`：最終選定嘗試的 Whisper ASR 辨識結果
- `cer`：最終選定嘗試的 CER 數值
- `seed`：最終選定嘗試使用的隨機種子
- `attempts`：實際嘗試的總次數
- `passed`：`true`（`CER <= cer_threshold`）或 `false`
- `attempt_history`：每次嘗試的詳細紀錄列表（`attempt`、`seed`、`cer`、`hypothesis`、`normalized_hyp`、`duration_sec`）

---

## 7. 執行單元測試

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/pytest /mnt/usr1/avi/Projects/formosan/Filtering/tests -v
```
