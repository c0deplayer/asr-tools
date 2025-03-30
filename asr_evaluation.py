from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger
from loguru_config import LoguruConfig

from src.asr_lib.config.eval_config import EvalConfig
from src.asr_lib.pipeline import EvaluationPipeline
from src.asr_lib.utils import read_config_ini

LoguruConfig.load("./src/asr_lib/config/logger/default.yaml")

if __name__ == "__main__":
    example_usage = """
Examples:
    # Run the complete pipeline with default configs
    python asr_evaluation.py --all

    # Generate hypotheses only with a custom runtime config
    python asr_evaluation.py --hyp_gen --eval-config my_config

    # Force re-running preprocessing and evaluation on existing hypotheses
    python asr_evaluation.py --preprocess --evaluation --force-prep-eval

    # Force re-running the entire pipeline
    python asr_evaluation.py --all --force-all
    """

    parser = argparse.ArgumentParser(
        description="ASR Evaluation Toolkit v0.0.1",
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

    pipeline_group = parser.add_argument_group(
        "Pipeline Options",
        description=(
            "Specify which part of the pipeline to run. "
            "Only one option can be selected."
        ),
    )

    pipeline_mode = pipeline_group.add_mutually_exclusive_group()
    pipeline_mode.add_argument(
        "--all",
        action="store_const",
        const="ALL",
        dest="pipeline",
        help=("Run the complete evaluation pipeline (default)"),
        default="ALL",
    )
    pipeline_mode.add_argument(
        "--hyp_gen",
        action="store_const",
        const="HYP_GEN",
        dest="pipeline",
        help="Generate ASR hypotheses only",
    )
    pipeline_mode.add_argument(
        "--preprocess",
        action="store_const",
        const="PREPROCESS",
        dest="pipeline",
        help="Prepare and format data only",
    )
    pipeline_mode.add_argument(
        "--evaluation",
        action="store_const",
        const="EVALUATION",
        dest="pipeline",
        help="Perform analysis on prepared data",
    )

    force_group = parser.add_argument_group(
        "Force Options",
        description=(
            "Control whether to force re-execution of pipeline components "
            "even if results already exist"
        ),
    )
    force_group.add_argument(
        "--force-all",
        action="store_true",
        help=(
            "Force execution of all pipeline stages "
            "(takes precedence over other force options)"
        ),
        default=False,
    )
    force_group.add_argument(
        "--force-hyp-gen",
        action="store_true",
        help="Force execution of the hypothesis generation step",
        default=False,
    )
    force_group.add_argument(
        "--force-prep",
        action="store_true",
        help="Force execution of the preprocessing step",
        default=False,
    )
    force_group.add_argument(
        "--force-eval",
        action="store_true",
        help="Force execution of the evaluation step",
        default=False,
    )

    args = parser.parse_args()

    logger.debug("Parsed command line arguments: {}", args)
    logger.info(
        "Starting ASR evaluation toolkit with pipeline mode: {}",
        args.pipeline,
    )

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    config_runtime_path = (
        Path("./src/asr_lib/config/eval") / f"{args.eval_config}.yaml"
    )
    if not config_runtime_path.exists():
        logger.error(
            "Runtime config file not found: {}. "
            "Please check the --eval-config parameter or create this file.",
            config_runtime_path,
        )
        sys.exit(1)

    logger.info("Loading runtime configuration: '{}'", args.eval_config)
    config_runtime = EvalConfig.from_yaml(config_runtime_path)
    logger.debug("Runtime configuration details: {}", config_runtime)

    config_user_path = (
        Path("./src/asr_lib/config/user") / f"{args.user_config}.ini"
    )
    if not config_user_path.exists():
        logger.error(
            "User config file not found: {}. "
            "Please check the --user-config parameter or create this file.",
            config_user_path,
        )
        sys.exit(1)

    config_user = read_config_ini(config_user_path)

    logger.info("Loading user configuration: '{}'", args.user_config)
    config_user = read_config_ini(config_user_path)
    logger.debug("User configuration details: {}", config_user)

    logger.debug("Initializing evaluation pipeline with loaded configurations")
    evaluation_pipeline = EvaluationPipeline(config_user, config_runtime)

    try:
        match args.pipeline:
            case "ALL":
                logger.info(
                    "▶️ Executing COMPLETE pipeline "
                    "(preprocessing → hypothesis generation → evaluation)",
                )

                force_options = {
                    "force_all": args.force_all,
                    "force_hyp_gen": args.force_hyp_gen,
                    "force_preprocess": args.force_prep,
                    "force_evaluation": args.force_eval,
                }
                if args.force_all:
                    logger.info(
                        "Force mode enabled: "
                        "All stages will be re-executed even if results exist",
                    )
                elif any(
                    [args.force_hyp_gen, args.force_prep, args.force_eval],
                ):
                    logger.info(
                        "Force mode enabled for specific stages: {}",
                        ", ".join(
                            k.replace("force_", "")
                            for k, v in force_options.items()
                            if v and k != "force_all"
                        ),
                    )

                evaluation_pipeline.evaluate(force_options=force_options)
                logger.success("✅ Complete pipeline execution finished")

            case "HYP_GEN":
                logger.info("▶️ Executing HYPOTHESIS GENERATION stage only")
                if args.force_hyp_gen:
                    logger.info(
                        "Force mode enabled: Existing hypotheses will be overwritten",
                    )

                evaluation_pipeline.generate_asr_hypothesis(
                    recreate_hypothesis=args.force_hyp_gen,
                )
                logger.success("✅ Hypothesis generation completed")

            case "PREPROCESS":
                logger.info("▶️ Executing PREPROCESSING stage only")
                if args.force_prep:
                    logger.info(
                        "Force mode enabled: Existing preprocessed data will be overwritten",
                    )

                evaluation_pipeline.prepare_asr_hypotheses(
                    force_preprocess=args.force_prep,
                )
                logger.success("✅ Preprocessing completed")

            case "EVALUATION":
                logger.info("▶️ Executing EVALUATION stage only")
                if args.force_eval:
                    logger.info(
                        "Force mode enabled: Metrics will be recalculated even if they exist",
                    )

                evaluation_pipeline.generate_metrics(
                    force_calculation=args.force_eval,
                )
                logger.success("✅ Evaluation completed")
            case _:
                msg = f"Invalid pipeline mode: {args.pipeline}"
                logger.error(msg)
                raise ValueError(msg)
    except KeyboardInterrupt:
        logger.warning(
            "❌ Pipeline execution interrupted by user. "
            "Partial results have been saved.",
        )
    except Exception as e:
        logger.exception("❌ Pipeline execution failed with error: {}", str(e))
        sys.exit(1)
