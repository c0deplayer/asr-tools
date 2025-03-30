from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, override

import librosa
import torch
from transformers import (
    SeamlessM4TForSpeechToText,
    SeamlessM4TProcessor,
    SeamlessM4Tv2ForSpeechToText,
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


class MetaSeamlessASR(BaseASR):
    # TODO: Better way to handle language codes,
    # so we do not need to hardcode every single language code.
    # Who wants to add manually more than 100 language codes
    LANGUAGE_CODES: ClassVar[dict[str, str]] = {
        "pl": "pol",
        "en": "eng",
        "de": "ger",
        "fr": "fre",
    }

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
        self.seamless_weights_cache_path = (
            self.model_weights_cache_path / f"hf_{model}"
            if self._use_huggingface
            else model
        )
        self.seamless_local_language = self.LANGUAGE_CODES[
            self.language_code.split("-")[0]
        ]

        if not initialize_model:
            self._logger.info("Skipping model initialization as requested...")
            self.seamless_processor = None
            self.seamless_model_local_default = None
            self.seamless_model_local_cpu = None
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
            self.seamless_processor, self.seamless_model_local_default = (
                self._get_model(token=token)
            )
        except Exception as e:
            self._logger.exception(
                "Failed to load default Whisper model: {}",
                e,
            )
            self._logger.exception("Using CPU instead...")
            self.device = get_torch_device()

            self.seamless_processor, self.seamless_model_local_default = (
                self._get_model()
            )

        if self.device.type != "cpu":
            self.seamless_model_local_cpu = self._get_model(
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

        processor = SeamlessM4TProcessor.from_pretrained(
            f"{self.provider}/{self.model}",
            token=token,
            cache_dir=self.seamless_weights_cache_path,
        )
        attn_implementation = get_attn_implementation(model_name=self.model)

        if "v2" in self.model:
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                f"{self.provider}/{self.model}",
                attn_implementation=attn_implementation,
                device_map=device,
                torch_dtype=torch.float32,  # For now hard-coded due to not supporting fp16
                use_safetensors=True,
                token=token,
                cache_dir=self.seamless_weights_cache_path,
            )
        else:
            model = SeamlessM4TForSpeechToText.from_pretrained(
                f"{self.provider}/{self.model}",
                attn_implementation=attn_implementation,
                device_map=device,
                torch_dtype=torch.float32,  # For now hard-coded due to not supporting fp16
                use_safetensors=True,
                token=token,
                cache_dir=self.seamless_weights_cache_path,
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

            inputs = self.seamless_processor(
                audios=speech_array,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )

            inputs = inputs.to(device=self.device, dtype=self.dtype)

            predicted_ids = self.seamless_model_local_default.generate(
                **inputs,
                tgt_lang=self.seamless_local_language,
            )

            asr_hyp = self.seamless_processor.batch_decode(
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

                inputs = self.seamless_processor(
                    speech_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    predicted_ids = self.seamless_model_local_cpu.generate(
                        **inputs,
                        tgt_lang=self.seamless_local_language,
                    )

                asr_hyp = self.seamless_processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True,
                )[0]

            except Exception:
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

        return asr_hyp
