from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import librosa
import polars as pl
from huggingface_hub import list_repo_files, snapshot_download
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.asr_lib.data.dataset import HFDataset
from src.asr_lib.metrics import load_metric
from src.asr_lib.models import load_asr_metadata, load_asr_system
from src.asr_lib.utils import (
    clear_gpu_memory,
    ensure_matching_attributes,
    find_correct_audio_path,
    get_torch_device,
)

from .functional.preprocessing import (
    NormalizationType,
    prepare_references_and_hyphoteses,
)

if TYPE_CHECKING:
    from src.asr_lib.config.eval_config import EvalConfig
    from src.asr_lib.models import BaseASR

console = Console()


class EvaluationPipeline:
    """Pipeline for evaluating ASR models against datasets.

    This class handles the entire evaluation process including downloading
    datasets and model weights, generating ASR hypotheses, preprocessing data,
    and calculating various metrics.

    Parameters
    ----------
    config_user : dict[str, Any]
        User configuration dictionary containing paths, tokens, and other settings.
    config_runtime : EvalConfig
        Runtime configuration object with details about models, datasets, and metrics.
    force_download : bool, default=False
        Whether to force re-download of datasets and model weights even if they exist.

    Attributes
    ----------
    config_runtime : EvalConfig
        Runtime configuration object.
    config_user : dict[str, Any]
        User configuration dictionary.
    metrics : list
        List of metric objects loaded based on config_runtime.metrics.

    """

    def __init__(
        self,
        config_user: dict[str, Any],
        config_runtime: EvalConfig,
        *,
        force_download: bool = False,
    ) -> None:
        """Initialize the evaluation pipeline.

        Parameters
        ----------
        config_user : dict[str, Any]
            User configuration dictionary containing paths, tokens, and other settings.
        config_runtime : EvalConfig
            Runtime configuration object with details about models, datasets, and metrics.
        force_download : bool, default=False
            Whether to force re-download of datasets and model weights even if they exist.

        """
        self.config_runtime = config_runtime
        self.config_user = config_user
        logger.info(
            "🚀 Initializing ASR evaluation pipeline with runtime config: '{}'",
            self.config_runtime.codename,
        )

        logger.debug(
            "Configuration includes {} ASR models, {} datasets, and {} metrics",
            len(self.config_runtime.asr_models),
            len(self.config_runtime.datasets),
            len(self.config_runtime.metrics),
        )

        logger.info(
            "📦 Preparing resources - downloading required datasets and models",
        )
        self._download_dataset_and_model_weights(force_download=force_download)

        self.metrics = [
            load_metric(metric) for metric in self.config_runtime.metrics
        ]

        logger.info(
            "✅ Evaluation pipeline initialized with {} metrics: {}",
            len(self.metrics),
            ", ".join(m.name for m in self.metrics),
        )

    def _download_dataset_and_model_weights(
        self,
        *,
        force_download: bool = False,
    ) -> None:
        """Download datasets and model weights from the HuggingFace Hub.

        This method handles downloading all required datasets and model weights for
        evaluation. For model weights, it intelligently downloads only .safetensors
        files when available.

        Parameters
        ----------
        force_download : bool, default=False
            If True, force re-download even if files already exist in cache.

        Returns
        -------
        None
            This method doesn't return anything but downloads necessary files to disk.

        Notes
        -----
        The method first downloads datasets and then models, prioritizing .safetensors
        files for models when available. It handles configuration for various model types
        and providers.

        """
        hf_token = self.config_user.get("TOKENS", {}).get("hf_token", None)
        cache_dir = Path(
            self.config_user.get("PATHS", {}).get(
                "common_cache_dir",
                f"{Path.cwd()}/.cache",
            ),
        )

        logger.info(
            "📦 Preparing resources - downloading required datasets and model weights{}",
            " (force download)" if force_download else "",
        )

        dataset_count = len(self.config_runtime.datasets)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            dataset_task = progress.add_task(
                "[green]Downloading datasets",
                total=dataset_count,
            )

            for dataset in self.config_runtime.datasets:
                progress.update(
                    dataset_task,
                    description=f"[green]Downloading dataset: {dataset.name}",
                )
                try:
                    HFDataset.download_only(
                        dataset_name=dataset.name,
                        subsets=dataset.subsets,
                        splits=dataset.splits,
                        cache_dir=cache_dir,
                        audio_path_col_name=dataset.audio_path_col_name,
                        audio_data_col_name=dataset.audio_data_col_name,
                        text_col_name=dataset.text_col_name,
                        hf_token=hf_token,
                        low_storage_usage=dataset.streaming,
                        max_samples_per_subset=dataset.max_samples_per_subset,
                    )
                    logger.success(
                        "✅ Dataset '{}' downloaded successfully",
                        dataset.name,
                    )
                except Exception as e:
                    logger.error(
                        "❌ Failed to download dataset '{}': {}",
                        dataset.name,
                        e,
                    )
                    logger.error(
                        "Please check your internet connection, HF token permissions, "
                        "and dataset name/availability.",
                    )
                    sys.exit(1)
                finally:
                    progress.update(dataset_task, advance=1)

        # Download model weights
        model_group_count = len(self.config_runtime.asr_models)
        logger.info(
            "🤖 Downloading {} model group{}",
            model_group_count,
            "s" if model_group_count > 1 else "",
        )

        for i, asr_model_group in enumerate(self.config_runtime.asr_models, 1):
            logger.info(
                "📥 Processing model group [{}/{}]: '{}'",
                i,
                model_group_count,
                asr_model_group.name,
            )

            if not asr_model_group.use_huggingface:
                logger.warning(
                    "⚠️  Model '{}' is not on HuggingFace. Pre-download is unavailable - "
                    "will download at runtime if needed.",
                    asr_model_group.name,
                )
                continue

            asr_model_params = self.config_runtime.get_specific_model_params(
                model_name=asr_model_group.name,
            )

            providers = ensure_matching_attributes(asr_model_group, "providers")
            versions = ensure_matching_attributes(asr_model_group, "versions")

            del asr_model_params["versions"]

            model_count = len(asr_model_group.models)
            for j, (model, provider, version) in enumerate(
                zip(
                    asr_model_group.models,
                    providers,
                    versions,
                    strict=True,
                ),
                1,
            ):
                model_id = f"{provider}/{model}"
                logger.info(
                    "🔄 Downloading model [{}/{}]: '{}' (version: {})",
                    j,
                    model_count,
                    model_id,
                    version,
                )

                try:
                    files = list_repo_files(
                        repo_id=model_id,
                        token=hf_token,
                    )
                    has_safetensors = any(
                        file.endswith(".safetensors") for file in files
                    )

                    model_cache_dir = (
                        cache_dir / f"model_weights/hf_{model}"
                        if asr_model_group.use_huggingface
                        else model
                    )
                    asr_model_params["version"] = version
                    asr_model_params["token"] = hf_token

                    ignore_patterns = ["*.msgpack", "*.h5"]
                    if has_safetensors:
                        ignore_patterns.append("*.pt")
                        ignore_patterns.append("*.bin")
                        logger.info(
                            "🔍 Found .safetensors files - optimizing download by skipping .pt/.bin files",
                        )
                    else:
                        logger.debug(
                            "No .safetensors files found in repository - downloading all weight formats",
                        )

                    logger.info(
                        "⬇️ Downloading model files to: {}{}",
                        model_cache_dir,
                        " (force download)" if force_download else "",
                    )
                    snapshot_download(
                        repo_id=model_id,
                        token=hf_token,
                        cache_dir=model_cache_dir,
                        ignore_patterns=ignore_patterns,
                        force_download=force_download,
                    )
                    logger.success(
                        "✅ Model '{}' (v{}) downloaded successfully",
                        model_id,
                        version,
                    )
                except Exception as e:
                    logger.error(
                        "❌ Failed to download model '{}': {}",
                        model_id,
                        e,
                    )
                    logger.error(
                        "Please check your internet connection, HF token permissions, "
                        "and model availability.",
                    )
                    sys.exit(1)

    def evaluate(
        self,
        *,
        force_options: dict[str, bool],
    ) -> None:
        """Execute the full evaluation pipeline.

        This method runs the three main stages of evaluation:
        1. Generate ASR hypotheses
        2. Prepare hypotheses and references for evaluation
        3. Calculate metrics and save results

        Parameters
        ----------
        force_options : dict[str, bool]
            Dictionary of options to control forced execution of each stage:
            - "force_all": Force all steps (overrides other options)
            - "force_hyp_gen": Force generation of ASR hypotheses
            - "force_preprocess": Force preprocessing of references/hypotheses
            - "force_evaluation": Force calculation of metrics


        """
        logger.info("📋 Starting complete evaluation pipeline with 3 stages")

        # Extract force options with clearer variable names
        force_all = force_options["force_all"]
        recreate_hypothesis = force_all or force_options["force_hyp_gen"]
        force_preprocess = force_all or force_options["force_preprocess"]
        force_calculation = force_all or force_options["force_evaluation"]

        if force_all:
            logger.info(
                "⚠️ Force mode enabled for ALL stages - existing results will be overwritten",
            )
        else:
            force_details = []
            if recreate_hypothesis:
                force_details.append("hypothesis generation")
            if force_preprocess:
                force_details.append("preprocessing")
            if force_calculation:
                force_details.append("metric calculation")

            if force_details:
                logger.info(
                    "⚠️ Force mode enabled for: {}",
                    ", ".join(force_details),
                )

        # Execute each pipeline stage
        logger.info("🎯 STAGE 1/3: Generating ASR hypotheses")
        self.generate_asr_hypothesis(recreate_hypothesis=recreate_hypothesis)
        logger.success("✅ ASR hypothesis generation completed")

        logger.info("🎯 STAGE 2/3: Preprocessing data for evaluation")
        self.prepare_asr_hypotheses(force_preprocess=force_preprocess)
        logger.success("✅ Data preprocessing completed")

        logger.info("🎯 STAGE 3/3: Calculating evaluation metrics")
        self.generate_metrics(force_calculation=force_calculation)
        logger.success("✅ Metric calculation completed")

        logger.success("🏆 Full evaluation pipeline executed successfully!")

    def generate_asr_hypothesis(
        self,
        *,
        recreate_hypothesis: bool = False,
    ) -> None:
        """Generate ASR hypotheses for audio samples in the datasets.

        This method loads ASR models and datasets defined in the configuration,
        processes the audio samples, and generates or retrieves ASR hypotheses.

        Parameters
        ----------
        recreate_hypothesis : bool, optional
            Whether to recreate hypotheses even if they already exist in cache,
            by default False

        Returns
        -------
        None
            This method doesn't return anything but creates hypothesis files
            for each audio sample that are saved to disk

        Raises
        ------
        NotImplementedError
            If the dataset is an IterableDataset, which is not fully supported


        """
        logger.info(
            "🔊 Generating ASR hypotheses{}",
            " (force recreation enabled)" if recreate_hypothesis else "",
        )

        hf_token = self.config_user.get("TOKENS", {}).get("hf_token", None)
        language_code = self.config_user.get("COMMON", {}).get(
            "LANG_CODE",
            "pl-PL",
        )
        common_cache_dir = self.config_user.get("PATHS", {}).get(
            "common_cache_dir",
            None,
        )

        for model_idx, asr_model_group in enumerate(
            self.config_runtime.asr_models,
            1,
        ):
            logger.info(
                "🤖 Processing ASR model group [{}/{}]: '{}'",
                model_idx,
                len(self.config_runtime.asr_models),
                asr_model_group.name,
            )

            asr_model_params = self.config_runtime.get_specific_model_params(
                model_name=asr_model_group.name,
            )

            logger.debug("Loading ASR model parameters: {}", asr_model_params)

            providers = ensure_matching_attributes(asr_model_group, "providers")
            versions = ensure_matching_attributes(asr_model_group, "versions")

            del asr_model_params["versions"]

            for model_instance_idx, (model, provider, version) in enumerate(
                zip(
                    asr_model_group.models,
                    providers,
                    versions,
                    strict=True,
                ),
                1,
            ):
                model_id = f"{provider}/{model}"
                logger.info(
                    "🔍 Loading model [{}/{}]: '{}' (version: {})",
                    model_instance_idx,
                    len(asr_model_group.models),
                    model_id,
                    version,
                )

                additional_params = {
                    "token": hf_token,
                    "version": version,
                    "common_cache_dir": common_cache_dir,
                }
                asr_model_params |= additional_params

                try:
                    asr_model = load_asr_system(
                        system=asr_model_group.name,
                        provider=provider,
                        model=model,
                        language_code=language_code,
                        model_config=asr_model_params,
                    )
                    logger.success(
                        "✅ Loaded model '{}' (codename: {})",
                        model_id,
                        asr_model.codename,
                    )
                except Exception as e:
                    logger.error(
                        "❌ Failed to load ASR model '{}': {}",
                        model_id,
                        e,
                    )
                    logger.error(
                        "Check model availability and configuration. "
                        "The model may require additional dependencies.",
                    )
                    sys.exit(1)

                for dataset_idx, dataset in enumerate(
                    self.config_runtime.datasets,
                    1,
                ):
                    logger.info(
                        "📊 Processing dataset [{}/{}]: '{}'",
                        dataset_idx,
                        len(self.config_runtime.datasets),
                        dataset.name,
                    )

                    dataset_name = dataset.name
                    if len(dataset_name.split("/")) > 1:
                        dataset_name = dataset_name.split("/")[-1]

                    asr_model.update_dataset_name(dataset_name)
                    logger.debug(
                        "Updated dataset context for model: {}",
                        dataset_name,
                    )

                    try:
                        logger.info("🔄 Initializing dataset handler")
                        hf_dataset = HFDataset(
                            dataset_name=dataset.name,
                            subsets=dataset.subsets,
                            splits=dataset.splits,
                            max_samples_per_subset=dataset.max_samples_per_subset,
                            streaming=dataset.streaming,
                            common_cache_dir=common_cache_dir,
                            hf_token=hf_token,
                            audio_path_col_name=dataset.audio_path_col_name,
                            text_col_name=dataset.text_col_name,
                        )
                        logger.debug("Dataset handler initialized successfully")
                    except Exception as e:
                        logger.error(
                            "❌ Failed to initialize dataset '{}': {}. Attempting force download...",
                            dataset.name,
                            e,
                        )
                        try:
                            hf_dataset = HFDataset(
                                dataset_name=dataset.name,
                                subsets=dataset.subsets,
                                splits=dataset.splits,
                                max_samples_per_subset=dataset.max_samples_per_subset,
                                streaming=dataset.streaming,
                                force_download=True,
                                common_cache_dir=common_cache_dir,
                                hf_token=hf_token,
                                audio_path_col_name=dataset.audio_path_col_name,
                                text_col_name=dataset.text_col_name,
                            )
                        except Exception as e2:
                            logger.error(
                                "❌ Force download also failed: {}. Cannot continue.",
                                e2,
                            )
                            sys.exit(1)

                    for subset in dataset.subsets:
                        if dataset.splits == "ALL":
                            splits = ("train", "validation", "test")
                        else:
                            splits = dataset.splits

                        for split in splits:
                            logger.info(
                                "🎯 Processing subset: '{}/{}' with model '{}'",
                                subset,
                                split,
                                asr_model.codename,
                            )
                            try:
                                hf_dataset_per_split = hf_dataset.get_subset(
                                    subset,
                                    split,
                                )
                            except KeyError:
                                continue

                            audio_paths = hf_dataset_per_split[
                                dataset.audio_path_col_name
                            ]

                            logger.debug(
                                "Found {} audio samples to process",
                                len(audio_paths),
                            )
                            audio_paths = find_correct_audio_path(
                                audio_paths,
                                base_cache_dir=common_cache_dir,
                            )

                            logger.info(
                                "🔊 Generating ASR hypotheses for {} audio files{}",
                                len(audio_paths),
                                " (recreating existing hypotheses)"
                                if recreate_hypothesis
                                else "",
                            )

                            generated_hyps = self._generate_asr_hypothesis_from_audio_samples(
                                audio_paths,
                                asr_model,
                                recreate_hypothesis=recreate_hypothesis,
                            )

                            logger.success(
                                "✅ Processed {}/{} subset: {} hypotheses generated/retrieved",
                                subset,
                                split,
                                len(generated_hyps),
                            )

                logger.info("🧹 Cleaning up model resources")
                del asr_model
                clear_gpu_memory(
                    device=get_torch_device(
                        use_gpu=asr_model_params["use_gpu"],
                    ),
                )
                logger.debug("Model resources released and GPU memory cleared")

    def prepare_asr_hypotheses(self, *, force_preprocess: bool = False) -> None:
        """Prepare ASR hypotheses for evaluation by preprocessing datasets.

        This method loads ASR models and datasets, and prepares evaluation input
        files for each dataset/subset/split combination. It creates TSV files
        containing references and hypotheses for later metric calculation.

        Parameters
        ----------
        force_preprocess : bool, default=False
            Whether to force preprocessing of data even if output files already exist.

        Returns
        -------
        None
            This method doesn't return anything but creates evaluation input files
            on disk that will be used by subsequent evaluation steps.

        """
        logger.info(
            "📋 Preparing evaluation data{}",
            " (force preprocessing enabled)" if force_preprocess else "",
        )

        hf_token = self.config_user.get("TOKENS", {}).get("hf_token", None)
        language_code = self.config_user.get("COMMON", {}).get(
            "LANG_CODE",
            "pl-PL",
        )
        common_cache_dir = self.config_user.get("PATHS", {}).get(
            "common_cache_dir",
            None,
        )
        common_eval_input_path = Path(
            self.config_user.get("PATHS", {}).get(
                "eval_input",
                f"{Path.cwd() / 'data/eval_input'}",
            ),
        )
        logger.debug(
            "Preprocessing configuration:\n"
            "  - Language code: {}\n"
            "  - Cache directory: {}\n"
            "  - Evaluation input path: {}",
            language_code,
            common_cache_dir,
            common_eval_input_path,
        )

        for model_idx, asr_model_group in enumerate(
            self.config_runtime.asr_models,
            1,
        ):
            logger.info(
                "🤖 Processing ASR model group [{}/{}]: '{}'",
                model_idx,
                len(self.config_runtime.asr_models),
                asr_model_group.name,
            )

            asr_model_params = self.config_runtime.get_specific_model_params(
                model_name=asr_model_group.name,
            )

            providers = ensure_matching_attributes(asr_model_group, "providers")
            versions = ensure_matching_attributes(asr_model_group, "versions")

            del asr_model_params["versions"]

            for model_instance_idx, (model, provider, version) in enumerate(
                zip(
                    asr_model_group.models,
                    providers,
                    versions,
                    strict=True,
                ),
                1,
            ):
                model_id = f"{provider}/{model}"
                logger.info(
                    "🔍 Loading model metadata [{}/{}]: '{}' (version: {})",
                    model_instance_idx,
                    len(asr_model_group.models),
                    model_id,
                    version,
                )

                additional_params = {
                    "token": hf_token,
                    "version": version,
                    "common_cache_dir": common_cache_dir,
                }
                asr_model_params |= additional_params
                try:
                    asr_model = load_asr_metadata(
                        system=asr_model_group.name,
                        provider=provider,
                        model=model,
                        language_code=language_code,
                        model_config=asr_model_params,
                    )
                    logger.success(
                        "✅ Loaded model metadata for '{}'",
                        asr_model.codename,
                    )
                except Exception as e:
                    logger.error("❌ Failed to load ASR model metadata: {}", e)
                    sys.exit(1)

                for dataset_idx, dataset in enumerate(
                    self.config_runtime.datasets,
                    1,
                ):
                    logger.info(
                        "📊 Processing dataset [{}/{}]: '{}'",
                        dataset_idx,
                        len(self.config_runtime.datasets),
                        dataset.name,
                    )

                    dataset_name = dataset.name
                    if len(dataset_name.split("/")) > 1:
                        dataset_name = dataset_name.split("/")[-1]

                    asr_model.update_dataset_name(dataset_name)

                    try:
                        logger.info("🔄 Initializing dataset handler")
                        hf_dataset = HFDataset(
                            dataset_name=dataset.name,
                            subsets=dataset.subsets,
                            splits=dataset.splits,
                            max_samples_per_subset=dataset.max_samples_per_subset,
                            streaming=dataset.streaming,
                            common_cache_dir=common_cache_dir,
                            hf_token=hf_token,
                            audio_path_col_name=dataset.audio_path_col_name,
                            text_col_name=dataset.text_col_name,
                        )
                    except Exception as e:
                        logger.error(
                            "❌ Failed to initialize dataset: {}. Attempting force download...",
                            e,
                        )
                        try:
                            hf_dataset = HFDataset(
                                dataset_name=dataset.name,
                                subsets=dataset.subsets,
                                splits=dataset.splits,
                                max_samples_per_subset=dataset.max_samples_per_subset,
                                streaming=dataset.streaming,
                                force_download=True,
                                common_cache_dir=common_cache_dir,
                                hf_token=hf_token,
                                audio_path_col_name=dataset.audio_path_col_name,
                                text_col_name=dataset.text_col_name,
                            )
                        except Exception as e2:
                            logger.error(
                                "❌ Force download also failed: {}. Cannot continue.",
                                e2,
                            )
                            sys.exit(1)

                    for subset in dataset.subsets:
                        if dataset.splits == "ALL":
                            splits = ("train", "validation", "test")
                        else:
                            splits = dataset.splits

                        for split in splits:
                            logger.info(
                                "🔄 Preprocessing data for subset: '{}/{}' with model '{}'",
                                subset,
                                split,
                                asr_model.codename,
                            )

                            dataset_codename = str.join(
                                "+",
                                [subset, split],
                            )
                            eval_input_path_per_split = (
                                common_eval_input_path
                                / dataset_name
                                / dataset_codename
                                / asr_model.codename
                                / version
                                / self.config_runtime.codename
                            )

                            eval_input_path_per_split.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            eval_input_file_per_split = (
                                eval_input_path_per_split
                                / f"eval_input_{self.config_runtime.codename}.tsv"
                            )

                            if (
                                not eval_input_file_per_split.exists()
                                or force_preprocess
                            ):
                                logger.info(
                                    "📝 Creating evaluation input file: {}",
                                    eval_input_file_per_split.relative_to(
                                        common_eval_input_path,
                                    ),
                                )

                                try:
                                    hf_dataset.prepare_eval_input(
                                        asr_model,
                                        subset_split=(subset, split),
                                        metadata_columns=dataset.metadata_columns,
                                        save_to=eval_input_file_per_split,
                                    )
                                    logger.success(
                                        "✅ Created evaluation input file for {}/{}",
                                        subset,
                                        split,
                                    )
                                except Exception as e:
                                    logger.error(
                                        "❌ Failed to prepare evaluation input: {}",
                                        e,
                                    )
                                    continue

                            else:
                                logger.info(
                                    "⏭️ Evaluation input file already exists for {}/{} - skipping",
                                    subset,
                                    split,
                                )

                logger.info("🧹Cleaning up model resources")
                del asr_model
                clear_gpu_memory(
                    device=get_torch_device(
                        use_gpu=asr_model_params["use_gpu"],
                    ),
                )
                logger.debug("Model resources released and GPU memory cleared")

    def generate_metrics(self, *, force_calculation: bool = False) -> None:
        """Calculate evaluation metrics for all prepared data.

        This method processes the prepared evaluation input files and calculates
        all configured metrics for each dataset, subset, and model combination.
        Results are saved both at the individual level and aggregated.

        Parameters
        ----------
        force_calculation : bool, default=False
            Whether to force calculation of metrics even if output files already exist.

        Returns
        -------
        None
            This method doesn't return anything but creates metric output files on disk.

        """
        logger.info(
            "📊 Generating evaluation metrics{}",
            " (force calculation enabled)" if force_calculation else "",
        )

        hf_token = self.config_user.get("TOKENS", {}).get("hf_token", None)
        language_code = self.config_user.get("COMMON", {}).get(
            "LANG_CODE",
            "pl-PL",
        )
        common_cache_dir = self.config_user.get("PATHS", {}).get(
            "common_cache_dir",
            None,
        )
        common_eval_input_path = Path(
            self.config_user.get("PATHS", {}).get(
                "eval_input",
                f"{Path.cwd() / 'data/eval_input'}",
            ),
        )
        common_eval_output_path = Path(
            self.config_user.get("PATHS", {}).get(
                "eval_output",
                f"{Path.cwd() / 'data/eval_output'}",
            ),
        )

        logger.debug(
            "Metric calculation configuration:\n"
            "  - Input path: {}\n"
            "  - Output path: {}\n"
            "  - Metrics: {}",
            common_eval_input_path,
            common_eval_output_path,
            ", ".join(m.name for m in self.metrics),
        )

        for dataset_idx, dataset in enumerate(self.config_runtime.datasets, 1):
            lf_results_per_dataset = pl.LazyFrame()
            logger.info(
                "📊 Processing dataset [{}/{}]: '{}'",
                dataset_idx,
                len(self.config_runtime.datasets),
                dataset.name,
            )

            dataset_name = dataset.name
            if len(dataset_name.split("/")) > 1:
                dataset_name = dataset_name.split("/")[-1]

            eval_output_path_per_dataset = (
                common_eval_output_path
                / dataset_name
                / self.config_runtime.codename
            )

            eval_output_path_per_dataset.mkdir(parents=True, exist_ok=True)

            current_date = datetime.datetime.now(tz=datetime.UTC)
            timestamp = current_date.strftime("%Y%m%d")
            eval_output_file_per_dataset = (
                eval_output_path_per_dataset
                / f"eval_output_per_dataset_all_{timestamp}.tsv"
            )
            eval_output_avg_file_per_dataset = (
                eval_output_path_per_dataset
                / f"avg_eval_output_per_dataset_all_{timestamp}.tsv"
            )

            logger.debug(
                "Output files:\n"
                "  - Detailed results: {}\n"
                "  - Average metrics: {}",
                eval_output_file_per_dataset.name,
                eval_output_avg_file_per_dataset.name,
            )

            try:
                hf_dataset = HFDataset(
                    dataset_name=dataset.name,
                    subsets=dataset.subsets,
                    splits=dataset.splits,
                    max_samples_per_subset=dataset.max_samples_per_subset,
                    streaming=dataset.streaming,
                    common_cache_dir=common_cache_dir,
                    hf_token=hf_token,
                    audio_path_col_name=dataset.audio_path_col_name,
                    text_col_name=dataset.text_col_name,
                )
            except Exception as e:
                logger.error(
                    "❌ Failed to initialize dataset: {}. Attempting force download...",
                    e,
                )
                try:
                    hf_dataset = HFDataset(
                        dataset_name=dataset.name,
                        subsets=dataset.subsets,
                        splits=dataset.splits,
                        max_samples_per_subset=dataset.max_samples_per_subset,
                        streaming=dataset.streaming,
                        force_download=True,
                        common_cache_dir=common_cache_dir,
                        hf_token=hf_token,
                        audio_path_col_name=dataset.audio_path_col_name,
                        text_col_name=dataset.text_col_name,
                    )
                except Exception as e2:
                    logger.error(
                        "❌ Force download also failed: {}. Cannot continue.",
                        e2,
                    )
                    sys.exit(1)

            for subset in dataset.subsets:
                if dataset.splits == "ALL":
                    splits = ("train", "validation", "test")
                else:
                    splits = dataset.splits

                for split in splits:
                    logger.info("🔄 Processing subset: '{}/{}'", subset, split)

                    try:
                        hf_dataset_per_split = hf_dataset.get_subset(
                            subset,
                            split,
                        )
                        df_hf_dataset_per_split_shape = (
                            hf_dataset_per_split.to_polars().shape
                        )
                        logger.debug(
                            "Dataset split contains {} samples with {} columns",
                            df_hf_dataset_per_split_shape[0],
                            df_hf_dataset_per_split_shape[1],
                        )
                    except KeyError:
                        continue

                    dataset_codename = str.join(
                        "+",
                        [subset, split],
                    )

                    for model_idx, asr_model_group in enumerate(
                        self.config_runtime.asr_models,
                        1,
                    ):
                        logger.info(
                            "🤖 Processing model group [{}/{}]: '{}'",
                            model_idx,
                            len(self.config_runtime.asr_models),
                            asr_model_group.name,
                        )

                        asr_model_params = (
                            self.config_runtime.get_specific_model_params(
                                model_name=asr_model_group.name,
                            )
                        )

                        providers = ensure_matching_attributes(
                            asr_model_group,
                            "providers",
                        )
                        versions = ensure_matching_attributes(
                            asr_model_group,
                            "versions",
                        )

                        del asr_model_params["versions"]

                        for model_instance_idx, (
                            model,
                            provider,
                            version,
                        ) in enumerate(
                            zip(
                                asr_model_group.models,
                                providers,
                                versions,
                                strict=True,
                            ),
                            1,
                        ):
                            model_id = f"{provider}/{model}"
                            logger.info(
                                "🔍 Processing model [{}/{}]: '{}' (version: {})",
                                model_instance_idx,
                                len(asr_model_group.models),
                                model_id,
                                version,
                            )

                            additional_params = {
                                "token": hf_token,
                                "version": version,
                                "common_cache_dir": common_cache_dir,
                            }
                            asr_model_params |= additional_params

                            asr_model = load_asr_metadata(
                                system=asr_model_group.name,
                                provider=provider,
                                model=model,
                                language_code=language_code,
                                model_config=asr_model_params,
                            )
                            logger.debug(
                                "Loaded ASR model metadata: {}",
                                asr_model.codename,
                            )

                            eval_input_path_per_split = (
                                common_eval_input_path
                                / dataset_name
                                / dataset_codename
                                / asr_model.codename
                                / version
                                / self.config_runtime.codename
                            )

                            eval_input_file_per_split = (
                                eval_input_path_per_split
                                / f"eval_input_{self.config_runtime.codename}.tsv"
                            )

                            eval_output_path_per_split = (
                                common_eval_output_path
                                / dataset_name
                                / dataset_codename
                                / asr_model.codename
                                / version
                                / self.config_runtime.codename
                            )
                            eval_output_file_per_split = (
                                eval_output_path_per_split
                                / f"eval_output_per_sample_{self.config_runtime.codename}.tsv"
                            )

                            eval_output_path_per_split.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            if not eval_input_file_per_split.exists():
                                logger.error(
                                    "❌ Input file not found: {}. Run preprocessing stage first.",
                                    eval_input_file_per_split,
                                )
                                continue

                            try:
                                logger.debug(
                                    "Loading evaluation input from: {}",
                                    eval_input_file_per_split,
                                )
                                lf_eval_input = pl.scan_csv(
                                    eval_input_file_per_split,
                                    separator="\t",
                                )
                            except Exception as e:
                                logger.error(
                                    "❌ Failed to read input file {}: {}",
                                    eval_input_file_per_split.name,
                                    e,
                                )
                                continue

                            if (
                                not eval_output_file_per_split.exists()
                                or force_calculation
                            ):
                                logger.info(
                                    "📊 Calculating metrics for {}/{} with model '{}'{}",
                                    subset,
                                    split,
                                    asr_model.codename,
                                    " (forced recalculation)"
                                    if force_calculation
                                    else "",
                                )

                                lf_eval_results_no_metadata = self._calculate_metrics_per_sample(
                                    lf_eval_input=lf_eval_input,
                                    dataset_name=dataset_name,
                                    subset_split=(subset, split),
                                    asr_model_codename=asr_model.codename,
                                    audio_path_col_name=dataset.audio_path_col_name,
                                    text_col_name=dataset.text_col_name,
                                    normalization_types=self.config_runtime.normalization_types,
                                )

                                lf_eval_results_per_split = (
                                    lf_eval_results_no_metadata.join(
                                        lf_eval_input,
                                        how="left",
                                        left_on="audio_path",
                                        right_on=dataset.audio_path_col_name,
                                        coalesce=True,
                                    )
                                )

                                lf_eval_results_per_split = (
                                    lf_eval_results_per_split.drop(
                                        [
                                            "audio_path",
                                            dataset.text_col_name,
                                            f"hyp_{asr_model.codename}",
                                            "codename",
                                        ],
                                    )
                                )

                                logger.info(
                                    "💾 Saving per-sample results to: {}",
                                    eval_output_file_per_split.name,
                                )
                                lf_eval_results_per_split.collect().write_csv(
                                    eval_output_file_per_split,
                                    separator="\t",
                                )
                                logger.success("✅ Metrics saved successfully")
                            else:
                                logger.info(
                                    "⏭️ Metrics already exist for {}/{} with model '{}' - loading from file",
                                    subset,
                                    split,
                                    asr_model.codename,
                                )
                                lf_eval_results_per_split = pl.scan_csv(
                                    eval_output_file_per_split,
                                    separator="\t",
                                )

                            if (
                                lf_results_per_dataset.limit(1)
                                .collect()
                                .is_empty()
                            ):
                                lf_results_per_dataset = (
                                    lf_eval_results_per_split
                                )
                                logger.debug(
                                    "Initialized combined results dataset",
                                )
                            else:
                                lf_results_per_dataset = pl.concat(
                                    [
                                        lf_results_per_dataset,
                                        lf_eval_results_per_split,
                                    ],
                                )
                                logger.debug(
                                    "Added split results to combined dataset",
                                )

                            logger.info("🧹Cleaning up model resources")
                            del asr_model
                            clear_gpu_memory(
                                device=get_torch_device(
                                    use_gpu=asr_model_params["use_gpu"],
                                ),
                            )
                            logger.debug(
                                "Model resources released and GPU memory cleared",
                            )

            logger.info(
                "💾 Saving combined results for dataset '{}' to: {}",
                dataset.name,
                eval_output_file_per_dataset.name,
            )
            try:
                df_results = lf_results_per_dataset.collect()
                df_results.write_csv(
                    eval_output_file_per_dataset,
                    separator="\t",
                )
                logger.success(
                    "✅ Combined results saved successfully ({} samples)",
                    len(df_results),
                )
            except Exception as e:
                logger.error("❌ Failed to save combined results: {}", e)

            # Calculate and save averages
            logger.info(
                "📊 Calculating average metrics across systems for dataset '{}'",
                dataset.name,
            )

            # TODO: Find a better way to handle this
            non_metric_columns = [
                "dataset",
                "subset",
                "split",
                "id",
                "ref_type",
                "norm_type",
                "system",
                "reference",
                "hypothesis",
                "provider",
                "version",
                "model",
                "audio_path",
                "audio_duration",
            ]
            if dataset.metadata_columns:
                non_metric_columns.extend(dataset.metadata_columns)

            all_lf_columns = lf_results_per_dataset.collect_schema().names()
            metric_columns = [
                col
                for col in all_lf_columns
                if col not in non_metric_columns
                and not col.startswith("resources_")
            ]

            resource_columns = [
                col for col in all_lf_columns if col.startswith("resources_")
            ]

            logger.debug(
                "Found {} metric columns and {} resource columns to aggregate",
                len(metric_columns),
                len(resource_columns),
            )

            # Create aggregation expressions for metrics with mean and std paired
            agg_expressions = [pl.len().alias("sample_count")]

            for metric in metric_columns:
                agg_expressions.append(
                    pl.col(metric).mean().alias(f"avg_{metric}"),
                )
                agg_expressions.append(
                    pl.col(metric).std().alias(f"std_{metric}"),
                )

            # Add resource column aggregations if they're numeric
            for res_col in resource_columns:
                try:
                    # Test if column can be averaged
                    lf_results_per_dataset.select(pl.col(res_col).mean())
                    agg_expressions.append(
                        pl.col(res_col).mean().alias(f"avg_{res_col}"),
                    )
                    agg_expressions.append(
                        pl.col(res_col).std().alias(f"std_{res_col}"),
                    )
                except Exception:
                    logger.debug(
                        "⚠️ Skipping average calculation for non-numeric column: {}",
                        res_col,
                    )

            lf_avg_metrics = (
                lf_results_per_dataset.group_by(
                    ["dataset", "subset", "split", "norm_type", "system"],
                )
                .agg(agg_expressions)
                .sort(["subset", "split", "norm_type"])
            )

            logger.info(
                "💾 Saving average metrics for dataset '{}' to: {}",
                dataset.name,
                eval_output_avg_file_per_dataset.name,
            )

            lf_avg_metrics.collect().write_csv(
                eval_output_avg_file_per_dataset,
                separator="\t",
            )

            logger.success(
                "✅ Average metrics saved successfully ({} unique combinations)",
                len(lf_avg_metrics.collect()),
            )

        logger.success("🏆 Evaluation metrics generated and saved successfully")

    def _generate_asr_hypothesis_from_audio_samples(
        self,
        audio_paths: list[str],
        asr_model: BaseASR,
        *,
        recreate_hypothesis: bool = False,
    ) -> list[str]:
        """Generate ASR hypotheses for a list of audio paths.

        This method processes each audio file through the ASR model to generate
        a transcription hypothesis. It tracks progress and reports status updates.

        Parameters
        ----------
        audio_paths : list[str]
            List of paths to audio files to process
        asr_model : BaseASR
            The ASR model to use for transcription
        recreate_hypothesis : bool, default=False
            Whether to regenerate hypotheses even if they exist in cache

        Returns
        -------
        list[str]
            List of ASR hypotheses corresponding to the input audio paths

        """
        asr_hypotheses = []
        total_files = len(audio_paths)

        logger.info(
            "🎧 Processing {} audio files with model '{}'{}",
            total_files,
            asr_model.codename,
            " (recreating existing hypotheses)" if recreate_hypothesis else "",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[green]Transcribing audio with {asr_model.codename}",
                total=total_files,
            )

            for audio_path in audio_paths:
                try:
                    asr_hyp = asr_model.process_audio(
                        audio_path,
                        recreate_hypothesis=recreate_hypothesis,
                    )
                    asr_hypotheses.append(asr_hyp)
                except Exception as e:
                    logger.error(
                        "❌ Error processing audio file {}: {}",
                        Path(audio_path).name,
                        str(e),
                    )
                    asr_hypotheses.append(
                        "",
                    )  # Add empty string for failed hypotheses

                progress.update(task, advance=1)

        success_count = len([h for h in asr_hypotheses if h])
        logger.success(
            "✅ Processed {}/{} audio files successfully ({}%)",
            success_count,
            total_files,
            int((success_count / total_files) * 100),
        )

        return asr_hypotheses

    def _calculate_metrics_per_sample(
        self,
        lf_eval_input: pl.LazyFrame,
        dataset_name: str,
        subset_split: tuple[str, str],
        asr_model_codename: str,
        audio_path_col_name: str,
        text_col_name: str,
        normalization_types: list[str],
    ) -> pl.LazyFrame:
        """Calculate metrics for each sample in the evaluation input.

        Parameters
        ----------
        lf_eval_input : pl.LazyFrame
            LazyFrame containing evaluation input data including reference and hypothesis texts.
        dataset_name : str
            Name of the dataset used for evaluation.
        subset_split : tuple[str, str]
            Tuple containing the split names for the evaluation input.
        asr_model_codename : str
            Codename of the ASR model used for generating hypotheses.
        audio_path_col_name : str
            Name of the column containing audio file paths.
        text_col_name : str
            Name of the column containing reference transcriptions.
        normalization_types : list[str]
            List of normalization types to apply to references and hypotheses.

        Returns
        -------
        pl.LazyFrame
            LazyFrame containing calculated metrics for each sample.

        """
        lf_eval_output_all = pl.LazyFrame()
        subset, split = subset_split
        dataset_name = dataset_name.split("/")[-1]

        logger.info(
            "📊 Calculating metrics for {}/{} with {} normalization types",
            subset,
            split,
            len(normalization_types),
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            norm_task = progress.add_task(
                "[green]Calculating metrics with different normalizations",
                total=len(normalization_types),
            )

            for normalization_type_str in normalization_types:
                normalization_type = NormalizationType(normalization_type_str)
                hyp_col_name = f"hyp_{asr_model_codename}"

                progress.update(
                    norm_task,
                    description=f"[green]Applying {normalization_type_str.upper()} normalization",
                )

                references, hypotheses, audio_paths, ids = (
                    prepare_references_and_hyphoteses(
                        lf_eval_input=lf_eval_input,
                        ref_col_name=text_col_name,
                        hyp_col_name=hyp_col_name,
                        audio_path_col_name=audio_path_col_name,
                        normalization_type=normalization_type,
                    )
                )

                logger.info("💯 Calculating metrics on {} samples", len(ids))

                def get_audio_duration(path: str) -> float:
                    """Calculate the duration of an audio file in seconds.

                    Parameters
                    ----------
                    path : str
                        Path to the audio file.

                    Returns
                    -------
                    float
                        Duration of the audio file in seconds, rounded to 2 decimal places.
                        Returns 0.0 if the duration cannot be determined.

                    Notes
                    -----
                    This function handles different versions of librosa API:
                    - For librosa < 0.10.0: Uses `path` parameter
                    - For librosa >= 0.10.0: Uses `filename` parameter

                    """
                    try:
                        if tuple(
                            map(int, librosa.__version__.split(".")[:2]),
                        ) < (
                            0,
                            10,
                        ):
                            return round(librosa.get_duration(path=path), 2)
                        return round(librosa.get_duration(filename=path), 2)
                    except Exception:
                        logger.warning(
                            "⚠️ Could not determine duration for audio file {}: {}",
                            Path(path).name,
                            str(e),
                        )
                        return 0.0

                lf_results = pl.LazyFrame(
                    {
                        "dataset": [dataset_name] * len(ids),
                        "subset": [subset] * len(ids),
                        "split": [split] * len(ids),
                        "id": ids,
                        "ref_type": [text_col_name] * len(ids),
                        "norm_type": [normalization_type_str] * len(ids),
                        "system": [asr_model_codename] * len(ids),
                        "reference": references,
                        "hypothesis": hypotheses,
                        "audio_path": audio_paths,
                    },
                )

                # Add audio duration information
                logger.debug(
                    "Calculating audio durations for {} files",
                    len(audio_paths),
                )
                lf_results = lf_results.with_columns(
                    pl.col("audio_path")
                    .map_elements(get_audio_duration, return_dtype=pl.Float64)
                    .alias("audio_duration"),
                )

                for metric in self.metrics:
                    metric.reset()

                metrics_columns = {metric.name: [] for metric in self.metrics}

                logger.debug("Calculating individual metrics for each sample")
                for ref, hyp in zip(references, hypotheses, strict=True):
                    for metric in self.metrics:
                        metric.update(hyp, ref)
                        metrics_columns[metric.name].append(metric.compute())

                lf_results = lf_results.with_columns(
                    [
                        pl.Series(name, values)
                        for name, values in metrics_columns.items()
                    ],
                )

                # Calculate and log aggregate metrics
                logger.info("📈 Computing aggregate metrics across all samples")
                for metric in self.metrics:
                    metric.reset()
                    metric.update(hypotheses, references)
                    aggregate_value = metric.compute()
                    logger.opt(colors=True).info(
                        "   <green>{}</green> ({}) = <bold>{:.2f}</bold>",
                        metric.name,
                        normalization_type_str.upper(),
                        aggregate_value,
                    )

                # Append to output
                if lf_eval_output_all.limit(1).collect().is_empty():
                    lf_eval_output_all = lf_results
                else:
                    lf_eval_output_all = pl.concat(
                        [lf_eval_output_all, lf_results],
                    )

                progress.update(norm_task, advance=1)

        return lf_eval_output_all
