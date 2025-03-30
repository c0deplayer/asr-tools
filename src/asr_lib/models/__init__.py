from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base_asr import BaseASR


def load_asr_system(
    system: str,
    provider: str,
    model: str,
    language_code: str,
    *,
    model_config: dict[str, Any],
) -> BaseASR:
    """Load and initialize the appropriate ASR model based on input parameters.

    Parameters
    ----------
    system : str
        The type of ASR system to load (e.g., "whisper_local", "nvidia_nemo").
    provider : str
        The provider of the ASR system.
    model : str
        The specific model name to use.
    language_code : str
        The language code for the ASR model.
    model_config : dict[str, Any]
        Configuration dictionary specifically for the model initialization.

    Returns
    -------
    BaseASR
        An initialized ASR model instance.

    Raises
    ------
    ValueError
        If the requested ASR model is not supported.

    """
    match system.lower():
        case "whisper_local":
            from .whisper_asr_local import WhisperASRLocal

            return WhisperASRLocal(
                provider,
                model,
                language_code,
                **model_config,
            )
        case "nvidia_nemo":
            from .nvidia_nemo_asr import NvidiaNemoASR

            return NvidiaNemoASR(
                provider,
                model,
                language_code,
                **model_config,
            )
        case "meta-w2v":
            from .meta_w2v_asr import MetaW2VASR

            return MetaW2VASR(
                provider,
                model,
                language_code,
                **model_config,
            )
        case "meta-seamless":
            from .meta_seamless_asr import MetaSeamlessASR

            return MetaSeamlessASR(
                provider,
                model,
                language_code,
                **model_config,
            )
        case _:
            msg = (
                f"Unsupported ASR provider: {provider}. Could not load ASR "
                f"system."
            )
            raise ValueError(msg)


def load_asr_metadata(
    system: str,
    provider: str,
    model: str,
    language_code: str,
    *,
    model_config: dict[str, Any] | None = None,
) -> BaseASR:
    """Load ASR model metadata without initializing the actual model weights.

    This function is useful when you only need access to model attributes
    or cached hypotheses, without the overhead of loading model weights.

    Parameters
    ----------
    system : str
        The type of ASR system to load (e.g., "whisper_local", "nvidia_nemo").
    provider : str
        The provider of the ASR system.
    model : str
        The specific model name to use.
    language_code : str
        The language code for the ASR model.
    model_config : dict[str, Any] | None, optional
        Configuration dictionary specifically for the model initialization.
        Default is None, which will be converted to an empty dict.

    Returns
    -------
    BaseASR
        A lightweight ASR model instance without initialized model weights.

    Raises
    ------
    ValueError
        If the requested ASR model is not supported.

    """
    if model_config is None:
        model_config = {}

    # Add initialize_model=False to config
    metadata_config = {**model_config, "initialize_model": False}

    return load_asr_system(
        system,
        provider,
        model,
        language_code,
        model_config=metadata_config,
    )
