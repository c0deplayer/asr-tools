from __future__ import annotations

import datetime
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from src.asr_lib.utils import (
    get_torch_device,
    get_torch_dtype_from_string,
)
from src.asr_lib.utils.wrappers import requires_model

if TYPE_CHECKING:
    from transformers import ProcessorMixin


# TODO: We should not initialize models, if all we need is the cache hypotheses
class BaseASR(ABC):
    """Base class for all Automatic Speech Recognition (ASR) systems.

    This abstract class defines the interface and common functionality for all ASR
    implementations. It manages system metadata, provides logging, and defines the
    required methods that all concrete ASR implementations must provide.

    Attributes
    ----------
    provider : str
        The provider of the ASR system (e.g., NVIDIA, OpenAI, Google).
    model : str
        The specific model name used by the ASR system.
    language_code : str
        The language code this ASR system is configured for (e.g., 'en-US').
    max_audio_length_in_seconds : int
        Maximum supported audio length in seconds.
    version : str
        Version identifier for the ASR system, typically in YearQN format.
    codename : str
        Machine-readable identifier composed of provider, model, and language.
    name : str
        Human-readable identifier for the ASR system.
    cache : dict
        Storage for caching ASR results.

    """

    def __init__(
        self,
        provider: str,
        model: str,
        language_code: str,
        *,
        use_gpu: bool = True,
        initialize_model: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the base ASR system.

        Parameters
        ----------
        provider : str
            The provider of the ASR system e.g. NVIDIA, OpenAI, etc.
        model : str
            The model used by the ASR system.
        language_code : str
            The language code of the ASR system (e.g., 'en-US', 'fr-FR').
        use_gpu : bool
            Whether to use GPU acceleration (default: True).
        initialize_model : bool
            Whether to initialize the model (default: True).
        kwargs : dict[str, Any] | None
            Additional keyword arguments. Supported options include:
            - max_audio_length_in_seconds: Maximum audio duration (default: 300)
            - version: Version string (default: current year and quarter)
            - dtype: Data type for model weights (default: 'float32')
            - common_cache_dir: Path to the common cache directory (default: ./cache)

        """
        dtype = kwargs.get("dtype", "float32")
        common_cache_dir = kwargs.get(
            "common_cache_dir",
            f"{Path.cwd()}/.cache",
        )

        self.provider = provider
        self.model = model
        self.language_code = language_code
        self.device = get_torch_device(use_gpu=use_gpu)
        self.dtype = get_torch_dtype_from_string(dtype)
        self.dataset_name = kwargs.get("dataset_name", "unknown")
        self.common_cache_path = Path(common_cache_dir)
        self.model_weights_cache_path = self.common_cache_path / "model_weights"
        self.dataset_eval_cache_path = (
            self.common_cache_path / f"asr_hypotheses/{self.dataset_name}"
        )
        self.max_audio_length_in_seconds = kwargs.get(
            "max_audio_length_in_seconds",
            300,
        )

        self.version = kwargs.get(
            "version",
            f"{datetime.datetime.now(tz=datetime.UTC).year}Q"
            f"{datetime.datetime.now(tz=datetime.UTC).month // 3 + 1}",
        )

        codename = (
            f"{self.provider.lower()}_{self.model.lower()}_"
            f"{self.language_code.lower()}"
        )
        self.codename = codename.replace("/", "_")
        self.name = f"{self.provider.upper()} - {self.model.upper()}"

        self._logger = logger.opt(colors=True)
        self._logger.info(
            "Initializing ASR | Provider <magenta>{}</magenta> | "
            "Model <magenta>{}</magenta> | Language <magenta>{}</magenta> | "
            "Version <magenta>{}</magenta>",
            self.provider,
            self.model,
            self.language_code,
            self.version,
        )

        self.cache_file_path = (
            self.dataset_eval_cache_path / f"{self.codename}.asr_cache.jsonl"
        )
        self.cache = {}

        self._model_initialized = initialize_model

    def load_cache_jsonl(self) -> None:
        """Load hypotheses cache from JSONL file."""
        self._logger.info(
            "Loading cache file for hypotheses: {}",
            self.cache_file_path,
        )
        if self.cache_file_path.exists():
            with self.cache_file_path.open() as f:
                for line in f:
                    self.cache.update(json.loads(line))
        else:
            self._logger.info(
                "Hypotheses cache file does not exist. Skipping...",
            )

    # TODO: Is there a better way to handle this?
    def update_dataset_name(self, new_dataset_name: str) -> None:
        """Update the dataset name and regenerate cache paths.

        Parameters
        ----------
        new_dataset_name : str
            The new dataset name to use.

        """
        self.dataset_name = new_dataset_name
        self.dataset_eval_cache_path = (
            self.common_cache_path / f"asr_hypotheses/{self.dataset_name}"
        )
        self.cache_file_path = (
            self.dataset_eval_cache_path / f"{self.codename}.asr_cache.jsonl"
        )

        self.cache = {}

        self.load_cache_jsonl()

    @requires_model
    def process_audio(
        self,
        audio_file: Path | str,
        *,
        recreate_hypothesis: bool = False,
    ) -> str:
        """Process audio data for ASR transcription.

        Parameters
        ----------
        audio_file : Path | str
            The audio file path to process.
        recreate_hypothesis : bool, optional
            Whether to recreate the hypothesis, by default False

        Returns
        -------
        str
            The processed audio result as text.

        """
        if not isinstance(audio_file, Path):
            audio_file = Path(audio_file)

        if not audio_file.exists():
            self._logger.warning(
                "Audio file does not exist: {}. Searching for alternative paths...",
                audio_file,
            )
            return ""

        # Check cache unless recreation is forced
        if not recreate_hypothesis:
            asr_hyp = self.get_value_from_cache(audio_file)
            if asr_hyp is not None and asr_hyp not in ("INVALID", "", "EMPTY"):
                return asr_hyp

            # if asr_hyp is None:
            #     self._logger.warning(
            #         "ASR hypothesis not found in cache. Generating...",
            #     )
            # elif asr_hyp in ("INVALID", "", "EMPTY"):
            #     self._logger.warning(
            #         "Found problematic ASR hypothesis: '{}'",
            #         asr_hyp,
            #     )
            # else:
            #     return asr_hyp

        # Start measuring time and VRAM usage
        start_time = time.time()
        initial_vram = self._get_initial_vram()

        # Generate hypothesis
        asr_hyp = self.generate_asr_hypothesis(audio_file)

        asr_hyp = asr_hyp.lstrip()

        # Calculate metrics
        elapsed_time = time.time() - start_time
        peak_vram, vram_diff = self._get_vram_metrics(initial_vram)

        # Handle empty hypothesis
        if asr_hyp == "":
            self._logger.warning("Empty ASR hypothesis. Saving as EMPTY")
            self.update_cache(
                audio_file,
                "EMPTY",
                elapsed_time=elapsed_time,
                peak_vram=peak_vram,
                vram_diff=vram_diff,
            )
            return "EMPTY"

        # Handle None hypothesis with retry
        if asr_hyp is None:
            self._logger.warning("None ASR hypothesis. Retrying once...")
            retry_start_time = time.time()
            asr_hyp = self.generate_asr_hypothesis(audio_file)
            elapsed_time += time.time() - retry_start_time

            # Update peak VRAM after retry
            new_peak_vram, _ = self._get_vram_metrics(initial_vram)
            if peak_vram is not None and new_peak_vram is not None:
                peak_vram = max(peak_vram, new_peak_vram)
            elif new_peak_vram is not None:
                peak_vram = new_peak_vram

            if not asr_hyp:
                status = "EMPTY" if asr_hyp == "" else "INVALID"
                self._logger.warning("ASR hypothesis is still {}", status)
                self.update_cache(
                    audio_file,
                    status,
                    elapsed_time=elapsed_time,
                    peak_vram=peak_vram,
                    vram_diff=vram_diff,
                )
                return status

        # Update cache with successful hypothesis
        self.update_cache(
            audio_file,
            asr_hyp,
            elapsed_time=elapsed_time,
            peak_vram=peak_vram,
            vram_diff=vram_diff,
        )

        return asr_hyp

    def _get_initial_vram(self) -> int | None:
        """Get initial VRAM usage based on device type.

        Returns
        -------
        int | None
            Initial VRAM usage in bytes, or None if not applicable.

        """
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated()
        if self.device.type == "mps":
            return torch.mps.current_allocated_memory()

        return None

    def _get_vram_metrics(
        self,
        initial_vram: int | None,
    ) -> tuple[int | None, int | None]:
        """Calculate peak VRAM and VRAM difference.

        Parameters
        ----------
        initial_vram : int | None
            Initial VRAM usage in bytes.

        Returns
        -------
        tuple[int | None, int | None]
            Peak VRAM usage and VRAM difference in bytes.

        """
        peak_vram = None
        vram_diff = None

        if self.device.type == "cuda":
            current_vram = torch.cuda.memory_allocated(device=None)
            peak_vram = torch.cuda.max_memory_allocated(device=None)
            if initial_vram is not None:
                vram_diff = current_vram - initial_vram

        elif self.device.type == "mps":
            current_vram = torch.mps.current_allocated_memory()
            peak_vram = torch.mps.driver_allocated_memory()
            if initial_vram is not None:
                vram_diff = current_vram - initial_vram

        return peak_vram, vram_diff

    @abstractmethod
    def generate_asr_hypothesis(self, audio_file: str | Path) -> str:
        """Generate an ASR hypothesis for the given audio.

        Parameters
        ----------
        audio_file : str | Path
            The audio file path to generate the hypothesis for.

        Returns
        -------
        str
            The generated ASR hypothesis (transcribed text).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        """
        msg = "Subclass must implement generate_asr_hypothesis method"
        raise NotImplementedError(msg)

    @abstractmethod
    def _get_model(
        self,
        token: str | None = None,
        *,
        use_huggingface: bool = False,
        device: torch.device | None = None,
    ) -> tuple[ProcessorMixin | None, torch.nn.Module]:
        """Get the ASR model from either the original source or HuggingFace.

        This method loads and returns the ASR model used for speech recognition.
        Subclasses must implement this method to specify how to fetch and
        initialize their specific model architecture.

        Parameters
        ----------
        token : str | None, optional
            The authentication token for accessing private models on HuggingFace.
            Required if use_huggingface is True and the model is private.
        use_huggingface : bool, optional
            Whether to load the model from HuggingFace instead of the original
            source. Default is False, which uses the model's native source.
        device : torch.device | None, optional
            The device (CPU/GPU/TPU) to load the model onto. If None, the model
            will be loaded onto the default device configured in the
            implementation's self.device attribute. Default is None.

        Returns
        -------
        tuple[ProcessorMixin | None, torch.nn.Module]
            A tuple containing the processor and the initialized ASR model ready for inference.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        """
        msg = "Subclass must implement get_model method"
        raise NotImplementedError(msg)

    def get_value_from_cache(
        self,
        audio_path: str | Path,
        *,
        key: str = "asr_hyp",
    ) -> str | None:
        """Retrieve the ASR hypothesis from the cache.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.
        key : str, optional
            Key to retrieve from the cache, by default "asr_hyp"

        Returns
        -------
        str | None
            The ASR hypothesis if found in the cache, otherwise None.

        """
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        return self.cache.get(audio_path, {}).get(self.version, {}).get(key)

    def update_cache(
        self,
        audio_path: str | Path,
        asr_hyp: str,
        *,
        elapsed_time: float | None = None,
        peak_vram: int | None = None,
        vram_diff: int | None = None,
    ) -> None:
        """Update the cache with a new hypothesis.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.
        asr_hyp : str
            The ASR hypothesis to cache.
        elapsed_time : float, optional
            Time taken to process the audio in seconds, by default None
        peak_vram : int, optional
            Peak VRAM usage in bytes, by default None
        vram_diff : int, optional
            VRAM difference before and after processing in bytes, by default None

        """
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        metadata = {
            "asr_hyp": asr_hyp,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "codename": self.codename,
            "hyp_gen_date": datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y%m%d",
            ),
        }

        if elapsed_time is not None:
            metadata["elapsed_time_seconds"] = round(elapsed_time, 3)
        if peak_vram is not None:
            metadata["peak_vram_bytes"] = peak_vram
        if vram_diff is not None:
            metadata["vram_diff_bytes"] = vram_diff

        self.cache[audio_path] = {self.version: metadata}

        # self._logger.info(
        #     "Updated cache for audio sample: <magenta>{}</magenta> with hypothesis: {}",
        #     audio_path.split("/")[-1],
        #     asr_hyp,
        # )

        # if elapsed_time is not None:
        #     self._logger.info(
        #         "Transcription took <magenta>{:.2f} seconds</magenta>",
        #         elapsed_time,
        #     )

        # if peak_vram is not None:
        #     self._logger.info(
        #         "Peak VRAM usage: <magenta>{:.2f} MB</magenta>",
        #         peak_vram / (1024 * 1024),
        #     )

        self.save_cache()

    def save_cache(self) -> None:
        """Save the current cache to disk in JSONL format."""
        self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)

        with self.cache_file_path.open("w") as f:
            for audio_path in self.cache:
                json.dump(
                    {audio_path: self.cache[audio_path]},
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    def get_cached_hypotheses(self) -> dict[str, Any]:
        """Get all cached hypotheses.

        Returns
        -------
        dict[str, Any]
            Dictionary of all cached hypotheses.

        """
        return self.cache
