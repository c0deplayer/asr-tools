from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import librosa
import torch

from src.asr_lib.utils import (
    get_attn_implementation,
    get_torch_device,
)
from src.asr_lib.utils.wrappers import requires_model

from .base_asr import BaseASR

if TYPE_CHECKING:
    from torch import nn
    from transformers import ProcessorMixin


class WhisperASRLocal(BaseASR):
    """Local implementation of Whisper Automatic Speech Recognition system.

    This class provides a local implementation of OpenAI's Whisper model for automatic
    speech recognition (ASR). It supports both HuggingFace and native Whisper implementations
    and can utilize GPU acceleration for faster inference.

    Parameters
    ----------
    provider : str
        The provider of the ASR system.
    model : str
        The specific Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
    language_code : str, optional
        The ISO language code for ASR. Defaults to "pl-PL" (Polish).
    sampling_rate : int, optional
        The audio sampling rate in Hz. Defaults to 16000.
    use_gpu : bool, optional
        Whether to use GPU acceleration for inference. Significantly improves
        performance for medium and larger models. Defaults to False.
    use_huggingface : bool, optional
        Whether to use the model from HuggingFace or using the OpenAI `whisper` package.
        Defaults to False.
    **kwargs : dict[str, Any]
        Additional keyword arguments passed to the BaseASR parent class.

    Attributes
    ----------
    device : torch.device
        The device (CPU/GPU) used for inference.
    sampling_rate : int
        The audio sampling rate used for processing.
    whisper_local_language : str
        The language code parsed for Whisper's internal use.
    whisper_processor : ProcessorMixin or None
        The processor for HuggingFace implementation.
    whisper_model_local_default : nn.Module
        The primary ASR model loaded on the specified device.
    whisper_model_local_cpu : nn.Module
        Fallback CPU model used when GPU inference fails.

    Notes
    -----
    For optimal performance, GPU usage is recommended especially for medium and larger models.
    The class includes fallback mechanisms to CPU when GPU inference fails.

    """

    def __init__(
        self,
        provider: str,
        model: str,
        language_code: str = "pl-PL",
        sampling_rate: int = 16000,
        *,
        use_gpu: bool = False,
        use_huggingface: bool = False,
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
        use_huggingface : bool, optional
            Whether to use HuggingFace for inference. Defaults to False.
        initialize_model : bool, optional
            Whether to initialize the model. Defaults to True.
        **kwargs : dict[str, Any]
            Additional keyword arguments for the BaseASR class.

        """
        # Before initializing the parent class, so the values
        # are not passed to the parent class
        hf_token = kwargs.pop("token", None)

        super().__init__(
            provider,
            model,
            language_code,
            use_gpu=use_gpu,
            initialize_model=initialize_model,
            **kwargs,
        )

        self.sampling_rate = sampling_rate
        self._use_huggingface = use_huggingface
        self.whisper_local_language = self.language_code.split("-")[0]
        self.whisper_weights_cache_path = self.model_weights_cache_path / (
            f"hf_{model}" if self._use_huggingface else model
        )

        if not initialize_model:
            self._logger.info("Skipping model initialization as requested...")
            self.whisper_processor = None
            self.whisper_model_local_default = None
            self.whisper_model_local_cpu = None
            return

        if hf_token is None and self._use_huggingface:
            self._logger.warning(
                "HuggingFace token not found in user_config. This may cause errors "
                "when accessing private models or models requiring acceptance of "
                "usage conditions.",
            )

        if not use_gpu or self.device.type == "cpu":
            self._logger.warning("Please use a GPU for better inference speed.")

        try:
            (
                self.whisper_processor,
                self.whisper_model_local_default,
            ) = self._get_model(token=hf_token, use_huggingface=use_huggingface)
        except Exception as e:
            self._logger.exception(
                "Failed to load default Whisper model: {}",
                e,
            )
            self._logger.exception("Using CPU instead...")
            self.device = get_torch_device()

            (
                self.whisper_processor,
                self.whisper_model_local_default,
            ) = self._get_model(token=hf_token, use_huggingface=use_huggingface)

        if self.device.type != "cpu":
            self.whisper_model_local_cpu = self._get_model(
                token=hf_token,
                device=get_torch_device(),
                use_huggingface=use_huggingface,
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

        if use_huggingface:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            processor = WhisperProcessor.from_pretrained(
                f"{self.provider}/{self.model}",
                token=token,
                cache_dir=self.whisper_weights_cache_path,
            )
            attn_implementation = get_attn_implementation(model_name=self.model)
            model = WhisperForConditionalGeneration.from_pretrained(
                f"{self.provider}/{self.model}",
                attn_implementation=attn_implementation,
                device_map=device,
                torch_dtype=self.dtype,
                use_safetensors=True,
                token=token,
                cache_dir=self.whisper_weights_cache_path,
            )
            model.generation_config.cache_implementation = "static"
            return processor, model

        import whisper

        return None, whisper.load_model(
            self.model,
            device=device,
            download_root=str(self.whisper_weights_cache_path),
        )

    def __generate_asr_hypothesis_using_hf(
        self,
        audio_file: str | Path,
        *,
        use_fallback: bool = False,
    ) -> str:
        """Generate ASR hypothesis using Whisper model from HuggingFace."""
        speech_array, sampling_rate = librosa.load(
            audio_file,
            sr=self.sampling_rate,
        )

        encodings = self.whisper_processor(
            audio=speech_array,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        model = (
            self.whisper_model_local_cpu
            if use_fallback
            else self.whisper_model_local_default
        )

        input_features = encodings.input_features.to(
            device=self.device,
            dtype=self.dtype,
        )

        predicted_ids = model.generate(
            input_features,
            language=self.whisper_local_language,
            task="transcribe",
        )

        return self.whisper_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

    def __generate_asr_hypothesis_using_local(
        self,
        audio_file: str | Path,
        *,
        use_fallback: bool = False,
    ) -> str:
        """Generate ASR hypothesis using the local Whisper model."""
        model = (
            self.whisper_model_local_cpu
            if use_fallback
            else self.whisper_model_local_default
        )

        if isinstance(audio_file, Path):
            audio_file = str(audio_file)

        return model.transcribe(
            audio_file,
            language=self.whisper_local_language,
        )["text"]

    @torch.inference_mode()
    @requires_model
    @override
    def generate_asr_hypothesis(self, audio_file: str | Path) -> str:
        generator = (
            self.__generate_asr_hypothesis_using_hf
            if self._use_huggingface
            else self.__generate_asr_hypothesis_using_local
        )

        try:
            asr_hyp = generator(audio_file)

        except Exception:
            if self.device.type == "cpu":
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

            self._logger.exception("Default model has failed. Using CPU...")

            try:
                asr_hyp = generator(audio_file, use_fallback=True)

            except Exception:
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

        return asr_hyp
