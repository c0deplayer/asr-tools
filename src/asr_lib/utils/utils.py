from __future__ import annotations

import configparser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path


def read_config_ini(
    config_path: Path | PathLike[str],
) -> dict[str, Any]:
    """Read a configuration file and return it as a nested dictionary.

    Parameters
    ----------
    config_path : Path or PathLike[str]
        Path to the configuration file to read

    Returns
    -------
    dict[str, Any]
        A nested dictionary containing the configuration data,
        with sections as the top-level keys

    """
    config = configparser.ConfigParser()
    config.read(config_path)

    return {section: dict(config[section]) for section in config.sections()}
