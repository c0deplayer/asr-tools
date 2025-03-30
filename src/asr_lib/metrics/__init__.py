from .base_metric import BaseMetric


def load_metric(name: str) -> BaseMetric:
    """Load a metric by name.

    Args:
        name: Name of the metric.

    Returns:
        The metric.

    """
    match name.upper():
        case "WER":
            from .wer import WordErrorRate

            return WordErrorRate()

        case "CER":
            from .cer import CharacterErrorRate

            return CharacterErrorRate()

        case "MER":
            from .mer import MatchErrorRate

            return MatchErrorRate()

        case "WIL":
            from .wil import WordInformationLoss

            return WordInformationLoss()

        case _:
            msg = f"Unknown metric '{name}'"
            raise ValueError(msg)
