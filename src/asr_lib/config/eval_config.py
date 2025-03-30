from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from os import PathLike


@dataclass(kw_only=True, frozen=True)
class ASRModelConfig:
    """ASR Model configuration.

    Parameters
    ----------
    name : str
        The name of the ASR system.
    models : list[str]
        List of model identifiers used by this ASR system.
    providers : list[str]
        List of providers used by this ASR system.
    versions : list[str]
        List of model versions used by this ASR system.
    use_huggingface : bool
        Whether to use HuggingFace models.

    """

    name: str
    models: list[str]
    providers: list[str]
    versions: list[str]
    use_huggingface: bool


@dataclass(kw_only=True, frozen=True)
class DatasetConfig:
    """Datasets configuration.

    Parameters
    ----------
    name : str
        The name of the dataset.
    subsets : list[str]
        List of dataset subsets to evaluate on.
    splits : list[str]
        List of data splits (e.g., 'train', 'dev', 'test') to evaluate on.
    audio_path_col_name : str
        Name of the audio path column in the dataset.
    audio_data_col_name : str
        Name of the audio data column in the dataset.
    text_col_name : str
        Name of the text column in the dataset.
    max_samples_per_subset : int
        Maximum number of samples to use from each subset.
    metadata_columns : list[str] | None = None
        List of metadata columns to save. Default is None.
    streaming : bool
        Whether to use streaming data loading.

    """

    name: str
    subsets: list[str]
    splits: list[str]
    audio_path_col_name: str
    audio_data_col_name: str
    text_col_name: str
    max_samples_per_subset: int
    metadata_columns: list[str] | None = None
    streaming: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataset configuration to a dictionary."""
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass(kw_only=True, frozen=True)
class EvalConfig:
    """Configuration class for evaluation settings.

    Parameters
    ----------
    name : str
        The human-readable name of the evaluation.
    codename : str
        The machine-readable identifier for the evaluation.
    metrics : list[str]
        List of metrics to compute.
    normalization_types : list[str]
        List of normalization types to apply.
    datasets : list[DatasetConfig]
        List of dataset configurations to evaluate.
    asr_systems : list[ASRSystemConfig]
        List of ASR systems configurations to evaluate.
    asr_system_params : dict[str, Any]
        Additional parameters for ASR systems.

    """

    name: str
    codename: str
    metrics: list[str]
    normalization_types: list[str]
    datasets: list[DatasetConfig]
    asr_models: list[ASRModelConfig]
    asr_model_params: dict[str, Any]

    @classmethod
    def from_yaml(cls, yaml_path: Path | PathLike[str]) -> EvalConfig:
        """Create an EvalConfig instance from a YAML configuration file.

        Parameters
        ----------
        yaml_path : Path or PathLike[str]
            Path to the YAML configuration file.

        Returns
        -------
        EvalConfig
            An instance of EvalConfig populated with the configuration from the
            YAML file.

        """
        if not isinstance(yaml_path, Path):
            yaml_path = Path(yaml_path)

        with yaml_path.open() as f:
            config = yaml.safe_load(f)

        config["asr_models"] = [
            ASRModelConfig(**system)
            for system in config.pop("asr_models", None)
        ]
        config["datasets"] = [
            DatasetConfig(**dataset) for dataset in config.pop("datasets", None)
        ]

        return cls(**config)

    def get_specific_model_params(self, model_name: str) -> dict[str, Any]:
        """Get the parameters for a specific ASR model.

        Parameters
        ----------
        model_name : str
            Name of the ASR model.

        Returns
        -------
        dict[str, Any]
            Parameters for the ASR model.

        """
        model_params = self.asr_model_params.copy()

        model_specific_params = {}
        for asr_model in self.asr_models:
            if asr_model.name == model_name:
                model_specific_params = {
                    "use_huggingface": asr_model.use_huggingface,
                    "versions": asr_model.versions,
                }
                break

        model_params |= model_specific_params

        return model_params
