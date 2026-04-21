import argparse
from argparse import Namespace

def parse_args():
    """
    Parses command line arguments.
    
    Returns:
        args (Namespace): Parsed command line arguments
    """
    
    # Create parser
    parser = argparse.ArgumentParser(description="Run MARL-SC experiments")

    # ----- Experiment type arg -----
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "tune", "evaluate", "seed-eval"],
        required=True,
        help="Experiment mode: 'single' for training, 'tune' for hyperparameter search, "
             "'evaluate' for evaluation, 'seed-eval' for multi-seed evaluation"
    )

    # ----- Config paths args -----
    parser.add_argument(
        "--env-config",
        type=str,
        help="Path to environment config YAML. Required for new single/tune runs; "
             "optional when resuming (loaded from saved experiment state)"
    )
    parser.add_argument(
        "--algorithm-config",
        type=str,
        help="Path to algorithm config YAML. Required for new single/tune runs; "
             "optional when resuming (loaded from saved experiment state)"
    )
    
    # ----- Output-related args -----
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="./experiment_outputs",
        help="Root directory for experiment outputs (experiment folders are created inside)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (used in folder structure). If not provided, auto-generated."
    )
    
    # ----- WandB args -----
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        help="WandB run name (for single mode)"
    )
    
    # ----- Resource-related args -----
    parser.add_argument(
        "--num-cpus",
        type=int,
        help="Number of CPUs per trial (tune mode)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs per trial (tune mode)"
    )
    parser.add_argument(
        "--num-cpus-per-env-runner",
        type=int,
        help="Number of CPUs per environment runner (tune mode)"
    )
    
    # ----- Other experiment-related args -----
    parser.add_argument(
        "--root-seed",
        type=int,
        help="Root seed for reproducibility. Split into independent train_seed and eval_seed internally."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from a previous run. "
             "In single mode: path to a checkpoint directory. "
             "In tune mode: experiment name"
    )

    # ----- Tune-specific args -----
    parser.add_argument(
        "--tune-config",
        type=str,
        help="Path to tune config YAML (required for new tune experiments)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of trials for tune mode (required for new tune experiments)"
    )

    # ----- Evaluation-specific args -----
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Path to checkpoint directory for evaluation. Mutually exclusive with --experiment-name."
    )
    parser.add_argument(
        "--checkpoint-number",
        type=str,
        help="Checkpoint identifier to evaluate (e.g. '50' for checkpoint_50, "
             "'000000' for checkpoint_000000). "
             "Only used with --experiment-name. If omitted, defaults to checkpoint_best "
             "(falls back to checkpoint_final if checkpoint_best does not exist)."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help="Number of evaluation episodes (overrides config default)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate visualization plots from manual rollout (evaluate mode only)"
    )

    # ----- Seed evaluation args -----
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds for seed evaluation (default: 5)"
    )
    parser.add_argument(
        "--tune-name",
        type=str,
        help="Name of a completed tune experiment for seed evaluation from tune results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top trials to save (tune mode) or evaluate (seed-eval mode). Default: 10"
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Fixed root seed for final evaluation in seed-eval mode (default: 42)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Override number of training iterations (seed-eval mode)"
    )

    # Parse arguments
    args = parser.parse_args()

    return validate_args(args)

def validate_args(args: Namespace):
    """
    Validates parsed command line arguments.
    
    Args:
        args (Namespace): Parsed command line arguments
    
    Returns:
        args (Namespace): Validated command line arguments
    """

    # Determine whether this is a resume run
    is_resume = args.resume_from is not None

    # ----- Validate required args per mode -----
    # Mode 'single' requires environment and algorithm config
    if args.mode == "single":
        if not is_resume:
            if not args.env_config:
                raise ValueError("--env-config is required for new single runs")
            if not args.algorithm_config:
                raise ValueError("--algorithm-config is required for new single runs")
    # Mode 'tune' requires environment, algorithm, and tune config, and number of samples
    elif args.mode == "tune":
        if not is_resume:
            if not args.env_config:
                raise ValueError("--env-config is required for new tune experiments")
            if not args.algorithm_config:
                raise ValueError("--algorithm-config is required for new tune experiments")
            if not args.tune_config:
                raise ValueError("--tune-config is required for new tune experiments")
            if args.num_samples is None:
                raise ValueError("--num-samples is required for new tune experiments")
    # Mode 'evaluate' requires checkpoint directory or experiment name
    elif args.mode == "evaluate":
        if args.checkpoint_dir and args.experiment_name:
            raise ValueError(
                "Cannot specify both --checkpoint-dir and --experiment-name. "
                "Use one or the other."
            )
        if not args.checkpoint_dir and not args.experiment_name:
            raise ValueError(
                "For evaluate mode, either --checkpoint-dir or --experiment-name "
                "(with optional --checkpoint-number) must be specified."
            )
        if args.checkpoint_number is not None and not args.experiment_name:
            raise ValueError(
                "--checkpoint-number can only be used with --experiment-name, "
                "not with --checkpoint-dir (which already points to a specific checkpoint)."
            )

    # Mode 'seed-eval' requires either tune-name or (env-config + algorithm-config)
    elif args.mode == "seed-eval":
        has_tune = args.tune_name is not None
        has_explicit = args.env_config is not None and args.algorithm_config is not None
        if not has_tune and not has_explicit:
            raise ValueError(
                "For seed-eval mode, provide either --tune-name (tune mode) or "
                "--env-config + --algorithm-config (single mode)"
            )
        if has_tune and has_explicit:
            raise ValueError(
                "Cannot specify both --tune-name and explicit configs. "
                "Use one or the other."
            )
        if has_explicit and not args.experiment_name:
            raise ValueError(
                "--experiment-name is required for single-mode seed evaluation"
            )

    return args