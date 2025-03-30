from __future__ import annotations

import sys
import time
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from datasets import (
    Dataset,
    DownloadMode,
    load_dataset,
)
from loguru import logger

from src.asr_lib.utils import find_correct_audio_path

if TYPE_CHECKING:
    from src.asr_lib.models.base_asr import BaseASR


class HFDataset:
    """A wrapper for Hugging Face datasets that handles caching and subset management.

    This class provides functionality to load and manage datasets from the Hugging Face
    datasets library, with support for different subsets and splits.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset on Hugging Face Hub.
    subsets : list[str]
        List of dataset subsets to load.
    splits : list[str]
        List of dataset splits to load (e.g., 'train', 'validation').
    max_samples_per_subset : int
        Maximum number of samples to load per subset.
    force_download : bool, optional
        Whether to force re-download the dataset, by default False.
    streaming : bool, optional
        Whether to load the dataset in streaming mode, by default False.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    subsets : list[str]
        List of dataset subsets.
    splits : list[str]
        List of dataset splits.
    max_samples_per_subset : int
        Maximum number of samples per subset.
    common_dataset_cache_path : Path
        Path to the cache directory for datasets.
    datasets : dict
        Dictionary to store loaded datasets.

    """

    def __init__(
        self,
        dataset_name: str,
        subsets: list[str],
        splits: list[str],
        max_samples_per_subset: int,
        *,
        force_download: bool = False,
        streaming: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the HFDataset instance.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset on Hugging Face Hub.
        subsets : list[str]
            List of dataset subsets to load.
        splits : list[str]
            List of dataset splits to load (e.g., 'train', 'validation').
        max_samples_per_subset : int
            Maximum number of samples to load per subset.
        force_download : bool, optional
            Whether to force re-download the dataset, by default False.
        streaming : bool, optional
            Whether to load the dataset in streaming mode, by default False.
        kwargs : dict[str, Any]
            Additional keyword arguments. Supported options include:
            - hf_token: Hugging Face token for accessing private datasets.
            - common_cache_dir: Common cache directory for datasets.
            - audio_path_col_name: Name of the audio column in the dataset containing audio paths.
            - text_col_name: Name of the text column in the dataset containing transcripts.

        """
        hf_token = kwargs.get("hf_token")

        self.dataset_name = dataset_name
        if "all" in subsets:
            self.subsets = ["train", "validation", "test"]
        else:
            self.subsets = subsets

        if hf_token is None:
            logger.warning(
                "⚠️ HuggingFace token not provided. Access to private datasets or datasets "
                "requiring usage condition acceptance may be restricted. Consider providing "
                "a token if you encounter authentication errors.",
            )

        self.splits = splits
        self.audio_path_col_name = kwargs.get("audio_path_col_name", "path")
        self.text_col_name = kwargs.get("text_col_name")
        self.max_samples_per_subset = max_samples_per_subset
        self.common_dataset_cache_path = Path(
            kwargs.get(
                "common_cache_dir",
                f"{Path.cwd()}/.cache",
            ),
        )
        self.datasets = {}

        logger.info(
            "📦 Initializing dataset: '{}' with {} subset(s) and {} split(s)",
            self.dataset_name,
            len(self.subsets),
            len(self.splits),
        )
        if max_samples_per_subset < sys.maxsize:
            logger.info(
                "⚙️ Sample limit active: Using max {} samples per subset",
                max_samples_per_subset,
            )

        start_time = time.time()
        self.__load_all_subsets(
            force_download=force_download,
            low_storage_usage=streaming,
            hf_token=hf_token,
        )
        logger.success(
            "✅ Dataset loaded successfully in {:.2f}s",
            time.time() - start_time,
        )

    # NOTE: This method will be implemented by subclasses to prepare
    # evaluation input data for ASR model assessment (That's for sure)
    def prepare_eval_input(
        self,
        asr_model: BaseASR,
        subset_split: tuple[str, str],
        *,
        metadata_columns: list[str] | None = None,
        save_to: Path | None = None,
    ) -> pl.DataFrame | None:
        """Prepare evaluation input data for ASR model assessment.

        This method prepares a DataFrame for ASR evaluation by extracting audio paths
        from the dataset, adding model metadata, reference columns, and retrieving
        ASR hypotheses from cache.

        Parameters
        ----------
        asr_model : BaseASR
            The ASR model to evaluate.
        subset_split : tuple[str, str]
            A tuple containing (subset, split) to identify which part of the dataset to use.
        metadata_columns : list[str] | None, optional
            Specific reference columns to include. If None, only column name defined
            in text_col_name will be used.
        save_to : Path | None, optional
            Path to save the resulting DataFrame as a TSV file. If provided, the method
            returns None instead of the DataFrame, by default None.

        Returns
        -------
        pl.DataFrame | None
            A DataFrame containing audio paths, model metadata, reference text, and
            ASR hypotheses. Returns None if save_to is specified.

        Raises
        ------
        NotImplementedError
            If the dataset is an IterableDataset, which is not fully supported.

        """
        subset, split = subset_split
        logger.info(
            "🔄 Preparing evaluation input for model '{}' using {}/{} subset",
            asr_model.codename,
            subset,
            split,
        )
        hf_dataset_per_split = self.get_subset(subset, split)

        logger.debug("Extracting audio paths from dataset")
        audio_paths = hf_dataset_per_split[self.audio_path_col_name]
        base_dataset_path = Path(self.common_dataset_cache_path) / "datasets"
        audio_paths = find_correct_audio_path(
            audio_paths,
            base_cache_dir=base_dataset_path,
        )

        logger.debug(
            "Creating evaluation LazyFrame with {} audio samples",
            len(audio_paths),
        )
        lf_eval_input = pl.LazyFrame({self.audio_path_col_name: audio_paths})
        lf_eval_input = lf_eval_input.with_columns(
            pl.lit(asr_model.provider).alias("provider"),
            pl.lit(asr_model.model).alias("model"),
            pl.lit(asr_model.version).alias("version"),
            pl.lit(asr_model.codename).alias("codename"),
        )

        logger.debug(
            "Available columns in dataset: {}",
            hf_dataset_per_split.column_names,
        )

        if metadata_columns is not None:
            ref_columns = [*metadata_columns, self.text_col_name]
        else:
            ref_columns = [self.text_col_name]

        logger.debug("Reference columns to be included: {}", ref_columns)
        for ref_col in ref_columns:
            if ref_col not in hf_dataset_per_split.column_names:
                logger.warning(
                    "⚠️ Reference column {} not found in dataset - skipping",
                    ref_col,
                )
                continue

            logger.opt(colors=True).debug(
                "➕ Adding reference column <magenta>{}</magenta>",
                ref_col,
            )
            lf_eval_input = lf_eval_input.with_columns(
                pl.Series(name=ref_col, values=hf_dataset_per_split[ref_col]),
            )

        logger.info(
            "🔍 Fetching ASR hypotheses and performance metrics from cache",
        )
        lf_eval_input = lf_eval_input.with_columns(
            [
                pl.col(self.audio_path_col_name)
                .map_elements(
                    lambda x: asr_model.get_value_from_cache(x),
                    return_dtype=pl.Utf8,
                )
                .alias(f"hyp_{asr_model.codename}"),
                pl.col(self.audio_path_col_name)
                .map_elements(
                    lambda x: asr_model.get_value_from_cache(
                        x,
                        key="elapsed_time_seconds",
                    ),
                    return_dtype=pl.Float64,
                )
                .alias("resources_gen_time_seconds"),
                pl.col(self.audio_path_col_name)
                .map_elements(
                    lambda x: asr_model.get_value_from_cache(
                        x,
                        key="peak_vram_bytes",
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("resources_peak_vram_bytes"),
                pl.col(self.audio_path_col_name)
                .map_elements(
                    lambda x: asr_model.get_value_from_cache(
                        x,
                        key="vram_diff_bytes",
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("resources_vram_diff_bytes"),
            ],
        )

        df_eval_input = lf_eval_input.collect()
        rows_count = len(df_eval_input)

        if save_to is not None:
            logger.info("💾 Saving evaluation data to: {}", save_to)
            df_eval_input.write_csv(file=save_to, separator="\t")
            logger.success(
                "✅ Evaluation data saved successfully ({} rows)",
                rows_count,
            )
            return None

        logger.success(
            "✅ Evaluation input prepared successfully ({} rows)",
            rows_count,
        )

        return df_eval_input

    def __load_all_subsets(
        self,
        *,
        force_download: bool = False,
        low_storage_usage: bool = False,
        hf_token: str | None = None,
    ) -> None:
        """Load all configured subsets and splits.

        Parameters
        ----------
        force_download : bool, optional
            Whether to force re-download the datasets, by default False.
        streaming : bool, optional
            Whether to load the datasets in streaming mode, by default False.
        hf_token : str | None, optional
            The Hugging Face token to use for authentication, by default None.

        """
        total_subsets = len(self.subsets) * len(self.splits)
        loaded = 0

        logger.info(
            "🔄 Loading {} dataset subset(s) across {} split(s) [{}mode]",
            len(self.subsets),
            len(self.splits),
            "low_storage_usage " if low_storage_usage else "",
        )

        if force_download:
            logger.warning(
                "⚠️ Force download enabled - all datasets will be redownloaded",
            )

        for subset in self.subsets:
            for split in self.splits:
                loaded += 1
                logger.info(
                    "📦 Loading subset {}/{}: '{}/{}' ({}/{})",
                    subset,
                    split,
                    self.dataset_name,
                    subset,
                    loaded,
                    total_subsets,
                )

                self.datasets[(subset, split)] = self._load_subset(
                    subset,
                    split,
                    force_download=force_download,
                    low_storage_usage=low_storage_usage,
                    hf_token=hf_token,
                )

    def _load_subset(
        self,
        subset: str,
        split: str,
        *,
        force_download: bool = False,
        low_storage_usage: bool = False,
        hf_token: str | None = None,
        trust_remote_code: bool = True,
    ) -> Dataset:
        """Load a specific subset and split of the dataset.

        Parameters
        ----------
        subset : str
            The subset to load.
        split : str
            The split to load.
        force_download : bool, optional
            Whether to force re-download the dataset, by default False.
        low_storage_usage : bool, optional
            Whether to save part of the dataset using streaming mode, by default False.
        hf_token : str | None, optional
            The Hugging Face token to use for authentication, by default None.
        trust_remote_code : bool, optional
            Whether to trust remote code, by default True.

        Returns
        -------
        Dataset  | IterableDataset:
            The loaded dataset subset.

        """
        cache_dir = str(
            self.common_dataset_cache_path
            / f"datasets/{self.dataset_name}-{subset}-{split}",
        )

        download_mode = (
            DownloadMode.FORCE_REDOWNLOAD
            if force_download
            else DownloadMode.REUSE_DATASET_IF_EXISTS
        )

        logger.info(
            "📥 Loading HuggingFace dataset: {} / {} / {}",
            self.dataset_name,
            subset,
            split,
        )
        if low_storage_usage:
            dataset = self.load_tsv_dataset(
                split=split,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        else:
            dataset = load_dataset(
                self.dataset_name,
                subset,
                split=split,
                streaming=False,
                cache_dir=cache_dir,
                download_mode=download_mode,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )

        selected_dataset = dataset.select(
            range(min(len(dataset), self.max_samples_per_subset)),
        )

        logger.debug(
            "Loaded dataset with {} samples (limited from {} total)",
            len(selected_dataset),
            len(dataset),
        )

        return selected_dataset

    def get_subset(
        self,
        subset: str,
        split: str,
    ) -> Dataset:
        """Get a specific subset and split from the loaded datasets.

        Parameters
        ----------
        subset : str
            The subset to retrieve.
        split : str
            The split to retrieve.

        Returns
        -------
        Dataset:
            The requested dataset subset.

        """
        if (subset, split) not in self.datasets:
            logger.error(
                "❌ Requested dataset subset not found: {} / {}. Available subsets: {}",
                subset,
                split,
                list(self.datasets.keys()),
            )
            msg = f"Dataset subset not found: {subset}/{split}"
            raise KeyError(msg)

        return self.datasets[(subset, split)]

    def load_tsv_dataset(
        self,
        split: str,
        cache_dir: str,
        *,
        force_download: bool = False,
    ) -> Dataset:
        """Load a TSV dataset from the cache directory.

        Parameters
        ----------
        split : str
            The split of the dataset to load (e.g., 'train', 'test', 'validation').
        cache_dir : str
            Path to the cache directory containing the dataset files.
        force_download : bool, optional
            Whether to force re-download the dataset, by default False.
            Currently not used in this implementation.

        Returns
        -------
        Dataset
            The loaded dataset from the TSV file.

        Raises
        ------
        SystemExit
            If the metadata file or audio clips directory is not found.

        """
        tsv_path = Path(cache_dir) / "metadata.tsv"
        audio_clips_path = Path(cache_dir) / "audio_clips"

        if not tsv_path.exists() or not audio_clips_path.exists():
            logger.error(
                "❌ Required files for TSV dataset not found:\n"
                "   - Metadata file exists: {}\n"
                "   - Audio clips directory exists: {}\n"
                "   Expected locations:\n"
                "   - {}\n"
                "   - {}\n"
                "   Run download_only() first or check permissions.",
                tsv_path.exists(),
                audio_clips_path.exists(),
                tsv_path,
                audio_clips_path,
            )
            sys.exit(1)

        logger.debug("Loading TSV dataset from: {}", tsv_path)
        return load_dataset(
            "csv",
            data_files={split: str(tsv_path)},
            split=split,
            cache_dir=cache_dir,
            delimiter="\t",
        )

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
        max_samples_per_subset: int = 0,
        low_storage_usage: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Download the dataset without loading it into memory.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to download.
        subsets : list[str]
            The subsets to download.
        splits : list[str]
            The splits to download.
        cache_dir : Path | PathLike[str]
            The directory where the dataset should be cached.
        hf_token : str | None, optional
            The token to use for Hugging Face authentication, by default None.
        trust_remote_code : bool, optional
            Whether to trust remote code, by default True.
        force_download : bool, optional
            Whether to force download even if the dataset is already cached, by default False.
        max_samples_per_subset : int, optional
            The maximum number of samples per split to download, by default 0.
        streaming : bool, optional
            Whether to download the dataset in streaming mode, by default False.
        kwargs : dict[str, Any]
            Additional keyword arguments to pass to the dataset constructor.

        """
        if hf_token is None:
            logger.warning(
                "⚠️  HuggingFace token not provided. You may encounter access restrictions "
                "for private datasets or datasets requiring explicit terms acceptance.",
            )

        if isinstance(cache_dir, (str, PathLike)):
            cache_dir = Path(cache_dir)

        download_mode = (
            DownloadMode.FORCE_REDOWNLOAD
            if force_download
            else DownloadMode.REUSE_DATASET_IF_EXISTS
        )

        logger.info(
            "🔽 Downloading dataset '{}' ({} subsets, {} splits){}",
            dataset_name,
            len(subsets),
            len(splits),
            " (force download)" if force_download else "",
        )

        if max_samples_per_subset > 0:
            logger.info(
                "⚙️ Download limited to {} samples per subset",
                max_samples_per_subset,
            )

        for subset in subsets:
            for split in splits:
                logger.info(
                    "📥 Downloading: {} / {} / {}",
                    dataset_name,
                    subset,
                    split,
                )
                cache_dir_per_split = str(
                    cache_dir / f"datasets/{dataset_name}-{subset}-{split}",
                )

                if low_storage_usage:
                    logger.debug(
                        "Using low storage download mode for dataset: {} / {} / {}",
                        dataset_name,
                        subset,
                        split,
                    )
                    cls._download_parts_of_dataset(
                        dataset_name=dataset_name,
                        subset=subset,
                        split=split,
                        cache_dir=cache_dir_per_split,
                        token=hf_token,
                        download_mode=download_mode,
                        trust_remote_code=trust_remote_code,
                        max_samples_per_subset=max_samples_per_subset,
                        **kwargs,
                    )
                else:
                    load_dataset(
                        dataset_name,
                        subset,
                        split=split,
                        streaming=False,
                        cache_dir=cache_dir_per_split,
                        token=hf_token,
                        download_mode=download_mode,
                        trust_remote_code=trust_remote_code,
                    )
                logger.debug(
                    "Completed download for: {} / {} / {}",
                    dataset_name,
                    subset,
                    split,
                )

        logger.success("✅ Successfully downloaded all dataset components")

    @staticmethod
    def _download_parts_of_dataset(
        dataset_name: str,
        subset: str,
        split: str,
        cache_dir: str,
        audio_data_col_name: str,
        audio_path_col_name: str,
        text_col_name: str,
        token: str | None = None,
        download_mode: DownloadMode = DownloadMode.REUSE_DATASET_IF_EXISTS,
        *,
        trust_remote_code: bool = False,
        max_samples_per_subset: int = 0,
    ) -> None:
        import polars as pl
        import soundfile as sf

        tsv_path = Path(cache_dir) / "metadata.tsv"

        if tsv_path.exists():
            logger.info(
                "📋 Metadata TSV already exists at {}. Skipping download...",
                tsv_path,
            )
            return

        audio_clips_path = Path(cache_dir) / "audio_clips"
        audio_clips_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "🔽 Streaming download of '{}' dataset ({}/{}) to {}",
            dataset_name,
            subset,
            split,
            audio_clips_path,
        )

        streaming_dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            token=token,
            download_mode=download_mode,
            trust_remote_code=trust_remote_code,
        )

        # TODO: Add seed as a parameter
        streaming_dataset = iter(streaming_dataset.shuffle(seed=42))

        batch_size = 100
        current_batch: list = []
        all_lazy_frames: list[pl.LazyFrame] = []
        count = 0

        for row in streaming_dataset:
            audio_col_parts = audio_data_col_name.split("-")
            audio_data = row[audio_col_parts[0]]
            audio_content = audio_data[audio_col_parts[1]]
            audio_path = audio_clips_path / audio_data.get(
                "path",
                f"{count}.wav",
            )
            sampling_rate = row.get(
                "sampling_rate",
                audio_data.get("sampling_rate", 16000),
            )

            if not Path(audio_path).parent.exists():
                Path(audio_path).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            audio_format = (
                "flac"
                if audio_path.suffix[1:] == "opus"
                else audio_path.suffix[1:]
            )

            sf.write(
                audio_path,
                audio_content,
                samplerate=sampling_rate,
                format=audio_format,
            )

            del row[audio_col_parts[0]]
            row[audio_path_col_name] = str(audio_path)

            current_batch.append(row)
            count += 1

            if len(current_batch) >= batch_size:
                # Convert batch to LazyFrame and keep reference
                all_lazy_frames.append(pl.LazyFrame(current_batch))
                current_batch = []
                logger.debug("Processed {} audio files", count)

            if max_samples_per_subset > 0 and count >= max_samples_per_subset:
                logger.debug(
                    "Reached sample limit ({}/{}) for {}/{}",
                    count,
                    max_samples_per_subset,
                    subset,
                    split,
                )
                break

        # Add any remaining items in the current batch
        if current_batch:
            all_lazy_frames.append(pl.LazyFrame(current_batch))

        # Combine all LazyFrames
        if all_lazy_frames:
            logger.info("📊 Creating metadata TSV with {} entries", count)

            # Concatenate all LazyFrames
            final_lf = pl.concat(all_lazy_frames)

            # Apply transcription transformations using Polars expressions
            final_lf = (
                final_lf.with_columns(
                    pl.col(text_col_name).str.replace_all('""', '"'),
                )
                .with_columns(
                    pl.when(
                        pl.col(text_col_name).str.starts_with('"')
                        & pl.col(text_col_name).str.ends_with('"'),
                    )
                    .then(
                        # Use regex to remove quotes only at beginning and end
                        pl.col(text_col_name).str.replace('^"(.*)"$', "$1"),
                    )
                    .otherwise(pl.col(text_col_name)),
                )
                .with_columns(
                    [
                        pl.when(
                            ~pl.col(text_col_name).str.ends_with(".")
                            & ~pl.col(text_col_name).str.ends_with("?")
                            & ~pl.col(text_col_name).str.ends_with("!"),
                        )
                        .then(pl.col(text_col_name) + ".")
                        .otherwise(pl.col(text_col_name))
                        .alias(text_col_name),
                    ],
                )
            )

            # Collect and write to file
            final_lf.collect().write_csv(tsv_path, separator="\t")

            logger.success(
                "✅ Successfully downloaded dataset to {} (TSV at {})",
                audio_clips_path,
                tsv_path,
            )
        else:
            logger.warning("No data was processed. Creating empty TSV file.")
            pl.DataFrame().write_csv(tsv_path, separator="\t")
