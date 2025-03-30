from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger
from loguru_config import LoguruConfig

from src.asr_lib.config.eval_config import EvalConfig
from src.asr_lib.stats.dataset_stats import DatasetStatsCalculator
from src.asr_lib.utils import read_config_ini

LoguruConfig.load("./src/asr_lib/config/logger/default.yaml")

if __name__ == "__main__":
    example_usage = """
Examples:
    # Generate dataset statistics with default configs
    python get_dataset_stats.py

    # Force re-running calculation
    python get_dataset_stats.py --force-calc

    """

    parser = argparse.ArgumentParser(
        description="Dataset Statistics Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
        epilog=f"{example_usage}\nFor more information, visit: https://github.com/your-repo/asr-tools",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        help=(
            "Name of the runtime configuration file without extension "
            "(default: test)"
        ),
        default="test",
        metavar="FILENAME",
    )

    parser.add_argument(
        "--user-config",
        type=str,
        help=(
            "Name of the user configuration file without extension "
            "(default: user_config)"
        ),
        default="user_config",
        metavar="FILENAME",
    )

    parser.add_argument(
        "--force-calc",
        action="store_true",
        help="Force execution of the hypothesis generation step",
        default=False,
    )
    args = parser.parse_args()

    logger.debug("Arguments: {}", args)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    config_runtime_path = (
        Path("./src/asr_lib/config/eval") / f"{args.eval_config}.yaml"
    )
    if not config_runtime_path.exists():
        logger.error(
            "Config file not found at {}. Please provide a valid path.",
            config_runtime_path,
        )
        sys.exit(1)

    config_runtime = EvalConfig.from_yaml(config_runtime_path)

    logger.info("Loading runtime configuration at {}", str(config_runtime_path))
    logger.debug("Runtime configuration: {}", config_runtime)

    config_user_path = (
        Path("./src/asr_lib/config/user") / f"{args.user_config}.ini"
    )
    if not config_user_path.exists():
        logger.error(
            "Config file not found at {}. Please provide a valid path.",
            config_user_path,
        )
        sys.exit(1)

    config_user = read_config_ini(config_user_path)

    logger.info("Loading user configuration at {}", str(config_user_path))
    logger.debug("User configuration: {}", config_user)

    stats_calculator = DatasetStatsCalculator(
        config_user=config_user,
        config_runtime=config_runtime,
    )
    stats_calculator.calculate_all_statistics()
    stats_calculator.save_statistics(force=args.force_calc)
    stats_calculator.print_summary()
