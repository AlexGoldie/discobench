"""A function so you can just import create_task and it will return make_files_public, make_files_private and task_description."""

from pathlib import Path
from typing import Any

import yaml

from discobench.utils.make_files import MakeFiles


def create_task(
    task_domain: str,
    test: bool,
    example: bool = False,
    use_base: bool = False,
    no_data: bool = False,
    config_path: str | None = None,
    config_dict: dict[str, Any] | None = None,
) -> None:
    """Prepare files for the training or testing subset of the task.

    Args:
        task_domain: The task domain to create the task for.
        test: Whether to create the train or test version of a task (as defined by the config).
        example: Whether to use the pre-built example task_config for the task_domain.
        use_base: Whether to use the baseline implementations for each editable module. Defaults to False, meaning a default task will use an `edit` implementation (i.e., only the interface for a module is defined).
        no_data: Whether to create the codebase without loading any of the data files. If the code loads a pretrained model, this will also be skipped.
        config_path: The path to the task configuration file. If not provided, the default task configuration file will be used. Check `discobench/tasks/{task_domain}/task_config.yaml` for expected structure for a given task.
        config_dict: A pre-built config dictionary, following the expected structure from `discobench/tasks/{task_domain}/task_config.yaml`.

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

    MakeFiles(task_domain).make_files(task_config, train=train, use_base=use_base, no_data=no_data)
