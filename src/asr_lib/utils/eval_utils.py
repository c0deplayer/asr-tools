from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger
from transformers.utils import (
    is_flash_attn_2_available,
    is_torch_sdpa_available,
)

if TYPE_CHECKING:
    from os import PathLike

    from src.asr_lib.config.eval_config import ASRModelConfig


def get_torch_device(*, use_gpu: bool = False) -> torch.device:
    """Return the torch device corresponding to the given device string.

    Parameters
    ----------
    use_gpu : bool, optional
        Whether to use GPU or not, by default False

    Returns
    -------
    torch.device
        The torch device corresponding to the given device string.

    """
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")

        logger.warning("No CUDA/MPS device available. Falling back to CPU...")
        return torch.device("cpu")

    return torch.device("cpu")


def get_attn_implementation(model_name: str) -> str:
    """Return the attention implementation to use.

    Parameters
    ----------
    model_name : str
        The name of the model to use.

    Returns
    -------
    str
        The attention implementation to use.

    """
    not_supported_models = ("w2v", "seamless")
    if any(model in model_name.lower() for model in not_supported_models):
        logger.warning(
            "⚠️ Model '{}' may not fully support optimized attention implementations. "
            "Defaulting to 'eager'. Performance might be suboptimal.",
            model_name,
        )
        return "eager"

    if is_flash_attn_2_available():
        logger.info("⚡ Flash Attention 2 detected. Using 'flash_attn_2'.")
        return "flash_attn_2"

    if is_torch_sdpa_available():
        return "sdpa"

    return "eager"


def get_torch_dtype_from_string(dtype: str) -> torch.dtype:
    """Get a torch dtype (e.g., `torch.float32`) from its string (e.g., `"float32"`)."""
    # It seems to be the only way for these dtypes
    torch_specific_dtypes = {
        "bfloat16": torch.bfloat16,
        "qint8": torch.qint8,
        "quint8": torch.quint8,
        "qint32": torch.qint32,
    }

    if dtype in torch_specific_dtypes:
        return torch_specific_dtypes[dtype]

    return dtype_numpy_to_torch(get_numpy_dtype_from_string(dtype))


def get_numpy_dtype_from_string(dtype: str) -> np.dtype:
    """Get a numpy dtype (e.g., `np.float32`) from its string (e.g., `"float32"`)."""
    return np.empty([], dtype=str(dtype).split(".")[-1]).dtype


def dtype_numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to its torch equivalent."""
    return torch.from_numpy(np.empty([], dtype=dtype)).dtype


def ensure_matching_attributes(
    config_group: ASRModelConfig,
    attribute_name: str,
) -> list[Any]:
    """Ensure the given attribute matches the number of items in the config group.

    Parameters
    ----------
    config_group : ASRModelConfig
        The configuration group containing the attribute to check.
    attribute_name : str
        The name of the attribute to check and possibly expand.

    Returns
    -------
    list[Any]
        The attribute values, one per expected item.

    Notes
    -----
    If there's only one item in the attribute but multiple entries expected, the attribute
    is duplicated for each entry. If there's a mismatch that can't be resolved,
    an error is logged and the program exits.

    """
    attribute_values = getattr(config_group, attribute_name)

    # Determine the expected count based on config type
    if hasattr(config_group, "models"):
        expected_count = len(config_group.models)
    else:
        msg = f"Unsupported config group type: {type(config_group)}"
        raise TypeError(msg)

    # Convert attribute_values to list if it's not already iterable
    if not hasattr(attribute_values, "__len__") or isinstance(
        attribute_values,
        str,
    ):
        attribute_values = [attribute_values]

    actual_count = len(attribute_values)

    # Handle expansion if needed
    if actual_count == 1 and expected_count > 1:
        return attribute_values * expected_count

    if actual_count != expected_count:
        logger.error(
            "❌ Attribute count mismatch in config group '{}': "
            "Attribute '{}' has {} values, but {} models were defined. "
            "Please provide one value per model or a single value to be applied to all.",
            config_group.name,
            attribute_name,
            actual_count,
            expected_count,
        )
        sys.exit(1)

    return attribute_values


def clear_gpu_memory(*, device: torch.device) -> None:
    """Clear GPU memory by deleting all tensors.

    This function iterates over all tensors in the current scope and deletes them.
    It's useful for freeing up GPU memory when working with large models or datasets.

    Parameters
    ----------
    device : torch.device
        The device to clear memory on.

    Notes
    -----
    This function should be called after each batch or epoch to ensure that GPU memory
    is not exhausted.

    """
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def find_correct_audio_path(
    audio_paths: list[str],
    base_cache_dir: PathLike[str] | Path,
) -> list[str]:
    """Find correct paths for audio files by verifying or searching for alternatives.

    Parameters
    ----------
    audio_paths : list[str]
        List of paths to audio files that need to be verified
    base_cache_dir : PathLike[str] or Path
        Base directory for the cache which may contain alternative locations

    Returns
    -------
    list[str]
        List of verified audio file paths, with empty strings for files that
        could not be found

    Notes
    -----
    The function employs two strategies for finding files:
    1. Check if the parent directory exists and contains files with the same name
    2. Search for matching files in the cache directory

    """
    verified_paths_map = {}
    unverified_paths_to_check = []

    # First check which paths exist already
    for audio_path_str in audio_paths:
        audio_file = Path(audio_path_str)
        if audio_file.exists():
            verified_paths_map[audio_path_str] = audio_file
        else:
            unverified_paths_to_check.append(audio_path_str)

    if not unverified_paths_to_check:
        return audio_paths

    logger.warning(
        "⚠️ {} audio paths do not exist. Searching for alternatives in cache: {}",
        len(unverified_paths_to_check),
        base_cache_dir,
    )

    if not isinstance(base_cache_dir, Path):
        base_cache_dir = Path(base_cache_dir)

    base_datasets_path = base_cache_dir / "datasets"

    # Process each unverified path
    for audio_path_str in unverified_paths_to_check:
        audio_file = Path(audio_path_str)
        found_alternative = False

        # Strategy 1: Check if parent directory exists and contains file with same name
        audio_path_parent = audio_file.parent
        if audio_path_parent.exists():
            for potential_match in audio_path_parent.rglob(
                f"{audio_file.name}*",
            ):
                logger.debug(
                    "Found alternative audio file: {}",
                    potential_match,
                )
                verified_paths_map[audio_path_str] = potential_match
                found_alternative = True
                break

        # Strategy 2: Search recursively in base cache/datasets directory if not found yet
        if not found_alternative and base_datasets_path.exists():
            filename = audio_file.name
            matches = list(
                base_datasets_path.rglob(filename),
            )  # Search for exact filename

            if matches:
                # Simple heuristic: choose the first match for now.
                # Could be improved by matching parent folder names if needed.
                best_match = matches[0]
                logger.debug(
                    "Found alternative via cache search ({} matches, using first): {}",
                    len(matches),
                    best_match,
                )
                verified_paths_map[audio_path_str] = best_match
                found_alternative = True

        if not found_alternative:
            logger.error(
                "❌ Could not find alternative path for: {}",
                audio_path_str,
            )
            verified_paths_map[audio_path_str] = None  # Mark as not found

    # Reconstruct the list in the original order
    final_paths = []
    for original_path in audio_paths:
        found_path = verified_paths_map.get(original_path)
        final_paths.append(str(found_path) if found_path else "")

    return final_paths
