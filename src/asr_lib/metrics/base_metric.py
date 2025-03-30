from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Abstract base class for implementing different metrics for ASR evaluation.

    This class defines the interface that all metrics should implement.
    Handles tracking of predictions and ground truth labels.

    Parameters
    ----------
    name : str
        The name of the metric.

    Attributes
    ----------
    name : str
        The name of the metric.
    pred_labels : list
        List of predicted labels.
    true_labels : list
        List of ground truth labels.
    state : dict
        Dictionary to store any additional state information needed for the metric.

    """

    def __init__(self, name: str) -> None:
        """Initialize the metric.

        Parameters
        ----------
        name : str
            The name of the metric.

        """
        self.name = name
        self.state: dict[str, Any] = {}

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state.

        This method should clear any accumulated statistics and reset the
        metric to its initial state.

        """

    @abstractmethod
    def update(
        self,
        pred_labels: list[str] | str,
        true_labels: list[str] | str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the metric with new predictions and ground truth.

        Parameters
        ----------
        pred_labels : list[str] or str
            Predicted labels, either as a list of strings or a single string.
        true_labels : list[str] or str
            Ground truth labels, either as a list of strings or a single string.
        metadata : dict[str, Any] or None, optional
            Additional metadata that may be used for metric calculation.

        """

    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value based on accumulated statistics.

        Returns
        -------
        float
            The computed metric value.

        """

    def __str__(self) -> str:
        """Return string representation of the metric."""
        try:
            value = self.compute()
        except Exception:
            return f"{self.name}: Not computed"
        else:
            return f"{self.name}: {value:.4f}"
