from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, override

import librosa
import torch
from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from src.asr_lib.utils import (
    get_attn_implementation,
    get_torch_device,
)
from src.asr_lib.utils.wrappers import requires_model

from .base_asr import BaseASR

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn
    from transformers import ProcessorMixin


class MetaW2VASR(BaseASR):
    def __init__(
        self,
        provider: str,
        model: str,
        language_code: str = "pl-PL",
        sampling_rate: int = 16000,
        *,
        use_gpu: bool = False,
        initialize_model: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the WhisperASRLocal class.

        Parameters
        ----------
        provider : str
            The provider of the ASR system.
        model : str
            The model to use for ASR.
        language_code : str, optional
            The language code to use for ASR. Defaults to "pl-PL".
        sampling_rate : int, optional
            The sampling rate to use for ASR. Defaults to 16000.
        use_gpu : bool, optional
            Whether to use a GPU for inference. Defaults to False.
        initialize_model : bool, optional
            Whether to initialize the model. Defaults to True.
        **kwargs : dict[str, Any]
            Additional keyword arguments for the BaseASR class.

        """
        token = kwargs.pop("token", None)

        super().__init__(
            provider,
            model,
            language_code,
            use_gpu=use_gpu,
            **kwargs,
        )

        self.sampling_rate = sampling_rate
        self._use_huggingface = True
        self.w2v_weights_cache_path = (
            self.model_weights_cache_path / f"hf_{model}"
            if self._use_huggingface
            else model
        )

        if not initialize_model:
            self._logger.info("Skipping model initialization as requested...")
            self.w2v_processor = None
            self.w2v_model_local_default = None
            self.w2v_model_local_cpu = None
            return

        if token is None and self._use_huggingface:
            self._logger.warning(
                "HuggingFace token not found in user_config. This may cause errors "
                "when accessing private models or models requiring acceptance of "
                "usage conditions.",
            )

        if not use_gpu or self.device.type == "cpu":
            self._logger.warning("Please use a GPU for better inference speed.")

        try:
            self.w2v_processor, self.w2v_model_local_default = self._get_model()
        except Exception as e:
            self._logger.exception(
                "Failed to load default Whisper model: {}",
                e,
            )
            self._logger.exception("Using CPU instead...")
            self.device = get_torch_device()

            self.w2v_processor, self.w2v_model_local_default = self._get_model()

        if self.device.type != "cpu":
            self.w2v_model_local_cpu = self._get_model(
                device=get_torch_device(),
            )[1]

    @override
    def _get_model(
        self,
        token: str | None = None,
        *,
        use_huggingface: bool = False,
        device: torch.device | None = None,
    ) -> tuple[ProcessorMixin | None, nn.Module]:
        if device is None:
            device = self.device

        attn_implementation = get_attn_implementation(model_name=self.model)

        if "bert" in self.model:
            processor = Wav2Vec2BertProcessor.from_pretrained(
                f"{self.provider}/{self.model}",
                token=token,
                cache_dir=self.w2v_weights_cache_path,
            )

            model = Wav2Vec2BertForCTC.from_pretrained(
                f"{self.provider}/{self.model}",
                attn_implementation=attn_implementation,
                device_map=device,
                torch_dtype=self.dtype,
                use_safetensors=True,
                token=token,
                cache_dir=self.w2v_weights_cache_path,
            )
        else:
            processor = Wav2Vec2Processor.from_pretrained(
                f"{self.provider}/{self.model}",
                token=token,
                cache_dir=self.w2v_weights_cache_path,
            )

            model = Wav2Vec2ForCTC.from_pretrained(
                f"{self.provider}/{self.model}",
                attn_implementation=attn_implementation,
                device_map=device,
                torch_dtype=self.dtype,
                use_safetensors=True,
                token=token,
                cache_dir=self.w2v_weights_cache_path,
            )

        return processor, model

    @torch.inference_mode()
    @requires_model
    @override
    def generate_asr_hypothesis(self, audio_file: str | Path) -> str:
        try:
            speech_array, sampling_rate = librosa.load(
                audio_file,
                sr=self.sampling_rate,
            )

            inputs = self.w2v_processor(
                speech_array,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )

            inputs = inputs.to(device=self.device, dtype=self.dtype)

            logits = self.w2v_model_local_default(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            asr_hyp = self.w2v_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]

        except Exception:
            if self.device.type == "cpu":
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

            self._logger.exception("Default model has failed. Using CPU...")

            try:
                speech_array, sampling_rate = librosa.load(
                    audio_file,
                    sr=self.sampling_rate,
                )

                inputs = self.w2v_processor(
                    speech_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    logits = self.w2v_model_local_cpu(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)

                asr_hyp = self.w2v_processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True,
                )[0]

            except Exception:
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

        return asr_hyp
