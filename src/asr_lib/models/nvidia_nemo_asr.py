from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, override

import librosa
import nemo.collections.asr as nemo_asr
import torch

from src.asr_lib.utils import get_torch_device
from src.asr_lib.utils.wrappers import requires_model

from .base_asr import BaseASR

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn
    from transformers import ProcessorMixin


class NvidiaNemoASR(BaseASR):
    """Local implementation of NVIDIA NeMo Automatic Speech Recognition system.

    This class provides a local implementation of NVIDIA's NeMo models for automatic
    speech recognition (ASR). It supports different model architectures like FastConformer
    and QuartzNet, and can utilize GPU acceleration for faster inference.

    Parameters
    ----------
    provider : str
        The provider of the ASR system.
    model : str
        The specific NeMo model to use.
    language_code : str, optional
        The ISO language code for ASR. Defaults to "pl-PL" (Polish).
    sampling_rate : int, optional
        The audio sampling rate in Hz. Defaults to 16000.
    use_gpu : bool, optional
        Whether to use GPU acceleration for inference. Significantly improves
        performance for complex models. Defaults to False.
    **kwargs : dict[str, Any]
        Additional keyword arguments passed to the BaseASR parent class.

    Attributes
    ----------
    device : torch.device
        The device (CPU/GPU) used for inference.
    sampling_rate : int
        The audio sampling rate used for processing.
    whisper_local_language : str
        The language code parsed for internal use.
    nemo_model_local_default : nn.Module
        The primary ASR model loaded on the specified device.
    nemo_model_local_cpu : nn.Module
        Fallback CPU model used when GPU inference fails.

    Notes
    -----
    For optimal performance, GPU usage is recommended especially for larger models.
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
        initialize_model: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the NvidiaNemoASR class.

        Parameters
        ----------
        provider : str
            The provider of the ASR system.
        model : str
            The model to use for ASR. It also supports HuggingFace for inference.
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
        super().__init__(
            provider,
            model,
            language_code,
            use_gpu=use_gpu,
            **kwargs,
        )

        self.sampling_rate = sampling_rate
        self._use_huggingface = False
        self.nemo_weights_cache_path = (
            self.model_weights_cache_path / f"hf_{model}"
            if self._use_huggingface
            else model
        )

        if not initialize_model:
            self._logger.info("Skipping model initialization as requested...")
            self.nemo_processor = None
            self.nemo_model_local_default = None
            self.nemo_model_local_cpu = None
            return

        if not use_gpu or self.device.type == "cpu":
            self._logger.warning("Please use a GPU for better inference speed.")

        if use_gpu and self.device.type != "cuda":
            self._logger.warning(
                "Nvidia NeMo models require a CUDA device for better inference speed.",
            )

        try:
            self.nemo_model_local_default = self._get_model()[1]
        except Exception as e:
            self._logger.exception(
                "Failed to load default Whisper model: {}",
                e,
            )
            self._logger.exception("Using CPU instead...")
            self.device = get_torch_device()

            self.nemo_model_local_default = self._get_model()[1]

        if self.device.type != "cpu":
            self.nemo_model_local_cpu = self._get_model(
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

        if "fastconformer" in self.model:
            return (
                None,
                nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                    model_name=self.model,
                    map_location=device,
                ),
            )

        if "quartznet" in self.model:
            return (
                None,
                nemo_asr.models.EncDecCTCModel.from_pretrained(
                    model_name=self.model,
                    map_location=device,
                ),
            )

        msg = f"Unknown model: {self.model}"
        self._logger.error(msg)
        raise ValueError(msg)

    @torch.inference_mode()
    @requires_model
    @override
    def generate_asr_hypothesis(self, audio_file: str | Path) -> str:
        try:
            speech_array, _ = librosa.load(
                audio_file,
                sr=self.sampling_rate,
            )

            result = self.nemo_model_local_default.transcribe(
                speech_array,
            )

            # For fastconformer models, result[0][0] is the hypothesis string
            # For quartznet models, result[0] is already the hypothesis string
            asr_hyp = (
                result[0][0] if "fastconformer" in self.model else result[0]
            )

        except Exception:
            if self.device.type == "cpu":
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

            self._logger.exception("Default model has failed. Using CPU...")

            try:
                result = self.nemo_model_local_cpu.transcribe(
                    speech_array,
                )

                asr_hyp = (
                    result[0][0] if "fastconformer" in self.model else result[0]
                )

            except Exception:
                self._logger.exception("Error generating ASR hypothesis")
                sys.exit()

        return asr_hyp
