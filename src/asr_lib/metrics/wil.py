from __future__ import annotations

from typing import Any

from nemo.collections.asr.metrics.wer import jiwer

from src.asr_lib.metrics.base_metric import BaseMetric


class WordInformationLoss(BaseMetric):
    """Word Information Loss metric for ASR evaluation.

    This class implements the Word Information Loss (WIL) metric, which measures the loss
    of information due to automatic speech recognition systems by comparing predicted transcriptions
    against reference transcriptions.

    Attributes
    ----------
    pred_labels : list
        List of predicted transcriptions.
    true_labels : list
        List of reference transcriptions.

    """

    def __init__(self, name: str = "WIL") -> None:
        """Initialize the WordInformationLoss metric.

        Parameters
        ----------
        name : str, optional
            Name of the metric, by default "WIL"

        """
        super().__init__(name=name)

        self.pred_labels = []
        self.true_labels = []

    def reset(self) -> None:
        """Reset the metric state.

        Clears all accumulated predictions and ground truth labels.
        """
        self.pred_labels.clear()
        self.true_labels.clear()

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
            Predicted transcriptions, either as a string or list of strings.
        true_labels : list[str] or str
            Reference transcriptions, either as a string or list of strings.
        metadata : dict[str, Any] or None, optional
            Additional metadata associated with the predictions, by default None.

        Notes
        -----
        If inputs are strings, they are converted to single-element lists.

        """
        if isinstance(pred_labels, str):
            pred_labels = [pred_labels]
        if isinstance(true_labels, str):
            true_labels = [true_labels]

        self.pred_labels.extend(pred_labels)
        self.true_labels.extend(true_labels)

    def compute(self) -> float:
        """Compute the metric value based on accumulated statistics.

        Returns
        -------
        float
            The Word Error Rate as a percentage, rounded to two decimal places.

        Notes
        -----
        Uses the JiWER library to compute Word Error Rate between accumulated
        reference and hypothesis transcriptions.

        """
        wil = jiwer.wil(
            reference=self.true_labels,
            hypothesis=self.pred_labels,
        )

        return round(wil * 100, 2)
