"""A function so you can just import create_task and it will return make_files_public, make_files_private and task_description."""

import argparse
from pathlib import Path
from typing import Any

import yaml

from discobench.utils.make_files import MakeFiles


def create_task(
    task_domain: str,
    test: bool,
    config_path: str | None = None,
    config_dict: dict[str, Any] | None = None,
    example: bool | None = None,
    no_data: bool | None = None,
) -> None:
    """Prepare files for the training or testing subset of the task.

    Args:
        task_domain: The task domain to create the task for.
        test: Whether to create the train or test version of a task (as defined by the config).
        config_path: The path to the task configuration file. If not provided, the default task configuration file will be used. Check `discobench/tasks/{task_domain}/task_config.yaml` for expected structure for a given task.
        config_dict: A pre-built config dictionary, following the expected structure from `discobench/tasks/{task_domain}/task_config.yaml`.
        example: Whether to use the pre-built example task_config for the task_domain.
        no_data: Whether to create the codebase without loading any of the data files. If the code loads a pretrained model, this will also be skipped.

    Notes:
        Only one of config_path, example OR config_dict (not more than one) should be passed as an argument here, to avoid any conflict.
    """
    explicit_configs = sum(arg is not None for arg in [config_path, config_dict])

    example_flag = 1 if example is True else 0

    if explicit_configs + example_flag > 1:
        raise ValueError("Provide only one of config_path, example=True, or config_dict.")

    if config_path is None and config_dict is None:
        if example is True:
            config_path = str(Path(__file__).parent / f"example_configs/{task_domain}.yaml")
        else:
            config_path = str(Path(__file__).parent / f"tasks/{task_domain}/task_config.yaml")
    if config_dict is not None:
        task_config = config_dict
    else:
        if config_path is None:
            raise ValueError("config_path cannot be None if config_dict is also None.")

        with open(config_path) as f:
            task_config = yaml.safe_load(f)

    train = not test

    MakeFiles(task_domain).make_files(task_config, train=train, no_data=no_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training task for a specified domain.")
    parser.add_argument(
        "--task_domain", type=str, default="OnPolicyRL", help="The task domain for algorithm discovery."
    )
    parser.add_argument("--test", action="store_true", default=False, help="If passed, create test task.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="The path to your task_config.yaml. If not provided, this will default to the DiscoBench task_config.yaml for your provided task_domain.",
    )
    parser.add_argument("--example", action="store_true", default=None, help="If passed, use example task config.")

    args = parser.parse_args()
    create_task(task_domain=args.task_domain, test=args.test, config_path=args.config_path, example=args.example)
