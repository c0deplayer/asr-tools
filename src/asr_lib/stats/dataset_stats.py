from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import librosa
import numpy as np
from datasets import load_dataset
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from src.asr_lib.config.eval_config import DatasetConfig, EvalConfig

type PathLikeStr = PathLike[str] | Path


class RunningStats:
    """Memory efficient class for calculating running statistics.

    Computes mean, variance, min, max without storing all values.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # Sum of squared differences from the mean
        self.min_val: float | None = None
        self.max_val: float | None = None

    def update(self, x: float) -> None:
        """Update statistics with a new value using Welford's algorithm."""
        self.n += 1

        # Update min and max
        if self.min_val is None or x < self.min_val:
            self.min_val = x
        if self.max_val is None or x > self.max_val:
            self.max_val = x

        # Update mean and variance using Welford's online algorithm
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self) -> dict[str, float]:
        """Get the current statistics."""
        if self.n < 1:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        std = np.sqrt(self.M2 / self.n) if self.n > 1 else 0.0
        return {
            "mean": self.mean,
            "std": std,
            "min": self.min_val if self.min_val is not None else 0.0,
            "max": self.max_val if self.max_val is not None else 0.0,
        }


@dataclass
class SplitStatistics:
    """Statistics for a specific dataset split."""

    audio_duration_stats: RunningStats = field(default_factory=RunningStats)
    transcript_length_stats: RunningStats = field(default_factory=RunningStats)
    sample_count: int = 0
    total_audio_seconds: float = 0.0

    @property
    def audio_hours(self) -> float:
        """Convert total audio seconds to hours."""
        return self.total_audio_seconds / 3600

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary for serialization."""
        return {
            "audio_hours": round(self.audio_hours, 2),
            "audio_duration_stats": {
                k: round(v, 2)
                for k, v in self.audio_duration_stats.get_stats().items()
            },
            "transcript_length_stats": {
                k: round(v, 2)
                for k, v in self.transcript_length_stats.get_stats().items()
            },
            "sample_count": self.sample_count,
        }


@dataclass
class DatasetStatistics:
    """Statistics container for speech datasets."""

    dataset_name: str
    overall_audio_duration_stats: RunningStats = field(
        default_factory=RunningStats,
    )
    overall_transcript_length_stats: RunningStats = field(
        default_factory=RunningStats,
    )
    split_stats: dict[str, SplitStatistics] = field(default_factory=dict)
    total_audio_seconds: float = 0.0

    @property
    def total_audio_hours(self) -> float:
        """Convert total audio seconds to hours."""
        return self.total_audio_seconds / 3600

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "total_audio_hours": round(self.total_audio_hours, 2),
            "audio_duration_stats": {
                k: round(v, 2)
                for k, v in self.overall_audio_duration_stats.get_stats().items()
            },
            "transcript_length_stats": {
                k: round(v, 2)
                for k, v in self.overall_transcript_length_stats.get_stats().items()
            },
            "split_stats": {
                split: stats.to_dict()
                for split, stats in self.split_stats.items()
            },
        }


class DatasetStatsCalculator:
    """Calculate statistics for speech datasets.

    This class calculates statistics about audio duration and transcript length
    for speech datasets defined in configuration files using memory-efficient methods.
    """

    def __init__(
        self,
        config_user: dict[str, Any],
        config_runtime: EvalConfig,
        output_path: PathLikeStr | None = None,
    ) -> None:
        self.cache_dir = Path(
            config_user.get(
                "common_cache_dir",
                f"{Path.cwd()}/.cache",
            ),
        )
        self.output_path = (
            Path(output_path) if output_path else Path.cwd() / "data/stats"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.hf_token = config_user.get("TOKENS", {}).get("hf_token")

        if self.hf_token is None:
            logger.warning(
                "HuggingFace token not found in user_config. This may cause errors "
                "when accessing private datasets or datasets requiring acceptance of "
                "usage conditions.",
            )

        self.statistics: dict[str, DatasetStatistics] = {}

        # Load configuration
        self.config_user = config_user
        self.config_runtime = config_runtime

    def _get_audio_duration(
        self,
        audio_data: np.ndarray,
        sampling_rate: int,
    ) -> float:
        """Calculate the duration of an audio file in seconds."""
        try:
            return round(
                librosa.get_duration(y=audio_data, sr=sampling_rate),
                2,
            )
        except Exception:
            logger.warning("Could not get duration")
            return 0.0

    def _get_transcript_length(self, transcript: str) -> int:
        """Get the length of a transcript in words."""
        if not transcript or not isinstance(transcript, str):
            return 0

        return len(transcript.split())

    def process_dataset(
        self,
        dataset_config: DatasetConfig,
    ) -> DatasetStatistics:
        dataset_name = dataset_config.name
        logger.opt(colors=True).info(
            "Processing dataset: <magenta>{}</magenta>",
            dataset_name,
        )

        stats = DatasetStatistics(dataset_name=dataset_name)

        for subset in dataset_config.subsets:
            for split in ("train", "validation", "test"):
                logger.opt(colors=True).info(
                    "Processing subset: <magenta>{}</magenta>, split: <magenta>{}</magenta>",
                    subset,
                    split,
                )

                # Initialize storage for this split
                split_key = f"{subset}-{split}"
                if split_key not in stats.split_stats:
                    stats.split_stats[split_key] = SplitStatistics()

                # Load dataset with streaming enabled
                try:
                    ds = load_dataset(
                        dataset_name,
                        subset,
                        split=split,
                        streaming=dataset_config.streaming,
                        cache_dir=str(self.cache_dir / "datasets"),
                        token=self.hf_token,
                    )
                except Exception:
                    logger.exception(
                        "Error loading dataset {}/{}/{}",
                        dataset_name,
                        subset,
                        split,
                    )
                    continue

                # Process each example
                audio_data_col_name = dataset_config.audio_data_col_name
                text_col = dataset_config.text_col_name
                split_stats = stats.split_stats[split_key]
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task(
                        f"Processing {split_key}",
                        total=None,
                    )

                    try:
                        total_samples = len(ds)
                        progress.update(task, total=total_samples)
                    except (TypeError, AttributeError):
                        # Dataset is a streaming dataset without known length
                        pass

                    for i, example in enumerate(ds):
                        # Get audio path and process it
                        try:
                            audio_col_parts = audio_data_col_name.split("-")
                            audio_data = example[audio_col_parts[0]]
                            audio_content = audio_data[audio_col_parts[1]]

                            sampling_rate = example.get(
                                "sampling_rate",
                                audio_data.get("sampling_rate", 16000),
                            )

                            if audio_content is not None:
                                duration = self._get_audio_duration(
                                    audio_content,
                                    sampling_rate=sampling_rate,
                                )
                                if duration > 0:
                                    split_stats.audio_duration_stats.update(
                                        duration,
                                    )
                                    stats.overall_audio_duration_stats.update(
                                        duration,
                                    )
                                    split_stats.total_audio_seconds += duration
                                    stats.total_audio_seconds += duration
                        except Exception:
                            logger.exception(
                                "Error processing audio at index {}",
                                i,
                            )

                        # Get transcript and process it
                        try:
                            transcript = example.get(text_col, "")
                            transcript_length = self._get_transcript_length(
                                transcript,
                            )

                            # Update running stats instead of appending to lists
                            split_stats.transcript_length_stats.update(
                                transcript_length,
                            )

                            # NOTE: BIGOS dataset does not provide transcript for test split,
                            # so we need to skip it to not skew the stats
                            if not (
                                "bigos" in dataset_name.lower()
                                and split == "test"
                                and transcript_length == 0
                            ):
                                stats.overall_transcript_length_stats.update(
                                    transcript_length,
                                )
                        except Exception:
                            logger.exception(
                                "Error processing transcript at index {}",
                                i,
                            )

                        if i % 10 == 0:  # Update progress every 10 items
                            progress.update(task, advance=10)

                        split_stats.sample_count += 1

        return stats

    def calculate_all_statistics(self) -> None:
        """Calculate statistics for all datasets in the configuration."""
        for dataset_config in self.config_runtime.datasets:
            dataset_name = dataset_config.name
            stats = self.process_dataset(dataset_config)
            self.statistics[dataset_name] = stats

    def save_statistics(
        self,
        file_format: str = "json",
        *,
        force: bool = False,
    ) -> None:
        if file_format != "json":
            logger.warning(
                "Unsupported format: {}, defaulting to JSON",
                file_format,
            )

        # Save individual dataset statistics
        for dataset_name, stats in self.statistics.items():
            safe_name = dataset_name.replace("/", "_")
            output_file = self.output_path / f"{safe_name}_statistics.json"

            if output_file.exists() and not force:
                logger.info(
                    "Statistics for <magenta>{}</magenta> already exist. Skipping...",
                    dataset_name,
                )
                continue

            with output_file.open(mode="w") as f:
                json.dump(stats.to_dict(), f, indent=2)

            logger.opt(colors=True).info(
                "Saved statistics for <magenta>{}</magenta> to <magenta>{}</magenta>",
                dataset_name,
                output_file,
            )

        # Save combined statistics
        combined_stats = {
            "datasets": {
                name: stats.to_dict() for name, stats in self.statistics.items()
            },
            "summary": {
                "total_datasets": len(self.statistics),
                "total_audio_hours": sum(
                    stats.total_audio_hours
                    for stats in self.statistics.values()
                ),
            },
        }

        combined_file = self.output_path / "combined_statistics.json"
        with combined_file.open(mode="w") as f:
            json.dump(combined_stats, f, indent=2)

        logger.info("Saved combined statistics to {}", combined_file)

    def print_summary(self) -> None:
        """Print a summary of the calculated statistics."""
        logger.info("===== Dataset Statistics Summary =====")

        for dataset_name, stats in self.statistics.items():
            logger.info("Dataset: {}", dataset_name)
            logger.info("  Total audio: {:.2f} hours", stats.total_audio_hours)

            audio_stats = stats.overall_audio_duration_stats.get_stats()
            logger.info(
                "  Average audio duration: {:.2f}s ± {:.2f}s",
                audio_stats.get("mean", 0),
                audio_stats.get("std", 0),
            )

            transcript_stats = stats.overall_transcript_length_stats.get_stats()
            logger.info(
                "  Average transcript length: {:.2f} ± {:.2f} words",
                transcript_stats.get("mean", 0),
                transcript_stats.get("std", 0),
            )

            logger.info("  Per-split statistics:")
            for split, split_stats in stats.split_stats.items():
                logger.info("    {}:", split)
                logger.info(
                    "      Samples: {}",
                    split_stats.sample_count,
                )
                logger.info(
                    "      Audio hours: {:.2f}",
                    split_stats.audio_hours,
                )

                audio_split_stats = split_stats.audio_duration_stats.get_stats()
                if audio_split_stats:
                    logger.info(
                        "      Avg duration: {:.2f}s ± {:.2f}s",
                        audio_split_stats.get("mean", 0),
                        audio_split_stats.get("std", 0),
                    )

                text_split_stats = (
                    split_stats.transcript_length_stats.get_stats()
                )
                if text_split_stats:
                    logger.info(
                        "      Avg transcript: {:.2f} ± {:.2f} words",
                        text_split_stats.get("mean", 0),
                        text_split_stats.get("std", 0),
                    )

            logger.info("")

        total_hours = sum(
            stats.total_audio_hours for stats in self.statistics.values()
        )
        logger.info("Total hours across all datasets: {:.2f}", total_hours)
