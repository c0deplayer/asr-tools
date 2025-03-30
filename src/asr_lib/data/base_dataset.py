from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    import polars as pl

    from src.asr_lib.models.base_asr import BaseASR


# TODO: Implement a template class for loading data that works
# for both HuggingFace and local file systems
class BaseDataset(ABC):
    @abstractmethod
    def prepare_eval_input(
        self,
        asr_model: BaseASR,
        subset_split: tuple[str, str],
        *,
        metadata_columns: list[str] | None = None,
        save_to: Path | None = None,
    ) -> pl.DataFrame | None:
        """Prepare evaluation input for the given subset and split.

        Parameters
        ----------
        asr_model : BaseASR
            The ASR model to use for preparing the input.
        subset_split : tuple[str, str]
            The subset and split to prepare input for.
        metadata_columns : list[str] | None, optional
            List of metadata columns to include in the output, by default None.
        save_to : Path | None, optional
            Path to save the output to, by default None.

        Returns
        -------
        pl.DataFrame | None
            The prepared evaluation input as a Polars DataFrame, or None if no data is available.

        """

    # TODO: Proper static typing
    @abstractmethod
    def get_subset(
        self,
        subset: str,
        split: str,
    ):
        """Get a subset of the dataset.

        Parameters
        ----------
        subset : str
            The subset to get.
        split : str
            The split to get.

        """

    @abstractmethod
    @classmethod
    def download_only(
        cls,
        dataset_name: str,
        subsets: list[str],
        splits: list[str],
        cache_dir: Path | PathLike[str],
        hf_token: str | None = None,
        *,
        trust_remote_code: bool = True,
        force_download: bool = False,
    ) -> None:
        """Download only the specified subsets and splits of the dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset on Hugging Face Hub.
        subsets : list[str]
            List of dataset subsets to download.
        splits : list[str]
            List of dataset splits to download.
        cache_dir : Path | PathLike[str]
            Directory to cache the dataset.
        hf_token : str | None, optional
            Hugging Face token for private datasets, by default None.
        trust_remote_code : bool, optional
            Whether to trust remote code, by default True.
        force_download : bool, optional
            Whether to force re-download the dataset, by default False.

        """
