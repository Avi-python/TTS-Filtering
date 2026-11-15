"""CosyVoice3 TTS engine wrapper with deterministic per-attempt seeding."""

from pathlib import Path
import sys
from typing import Literal

import torch


def _ensure_cosyvoice_paths(cosyvoice_repo: Path) -> None:
    """Add CosyVoice and Matcha-TTS directories to sys.path if not already present."""
    repo_str = str(cosyvoice_repo)
    matcha_str = str(cosyvoice_repo / "third_party" / "Matcha-TTS")
    for path_str in (matcha_str, repo_str):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


class TTSEngine:
    """Wrapper around CosyVoice3 AutoModel for seeded speech synthesis."""

    def __init__(
        self,
        model_dir: Path,
        cosyvoice_repo: Path,
        mode: Literal["zero_shot", "instruct2"] = "zero_shot",
        instruct_prefix: str = "You are a helpful assistant.<|endofprompt|>",
    ) -> None:
        _ensure_cosyvoice_paths(Path(cosyvoice_repo))
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.common import set_all_random_seed

        self._set_seed = set_all_random_seed
        self.model_dir = Path(model_dir)
        self.mode = mode
        self.instruct_prefix = instruct_prefix
        self.model = AutoModel(model_dir=str(self.model_dir))
        self.sample_rate: int = int(self.model.sample_rate)

    def synthesize(
        self,
        text: str,
        seed: int,
        prompt_wav: str,
        prompt_text: str | None = None,
        instruct_text: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Synthesize speech for `text` using `seed` and return `(waveform, sample_rate)`."""
        if not prompt_wav:
            raise ValueError("prompt_wav is required for CosyVoice3 synthesis")

        self._set_seed(seed)

        if self.mode == "zero_shot":
            if prompt_text is None:
                raise ValueError("prompt_text is required for zero_shot mode")
            full_prompt = (
                prompt_text
                if "<|endofprompt|>" in prompt_text
                else f"{self.instruct_prefix}{prompt_text}"
            )
            outputs = list(
                self.model.inference_zero_shot(
                    text,
                    full_prompt,
                    prompt_wav,
                    stream=False,
                    text_frontend=False,
                )
            )
        elif self.mode == "instruct2":
            full_instruct = instruct_text or self.instruct_prefix
            outputs = list(
                self.model.inference_instruct2(
                    text,
                    full_instruct,
                    prompt_wav,
                    stream=False,
                    text_frontend=False,
                )
            )
        else:
            raise ValueError(f"Unsupported TTS mode: {self.mode}")

        if not outputs:
            raise RuntimeError(f"CosyVoice3 returned no audio chunks for text: {text!r}")

        waveform = torch.cat([out["tts_speech"].cpu() for out in outputs], dim=-1)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, self.sample_rate
