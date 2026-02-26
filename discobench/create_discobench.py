"""A function to create a discobench task given the task_name. A list of task names can got from discobench.utils.get_discobench_tasks."""

from pathlib import Path

import yaml

from discobench.utils.make_files import MakeFiles


def create_discobench(task_name: str, test: bool, use_base: bool = False, no_data: bool = False) -> None:
    """Prepare files for the training or testing subset of a discobench task.

    Args:
        task_name: The DiscoBench task name.
        test: Whether to create the train or test version of a task (as defined by the config).
        use_base: Whether to use the baseline implementations for each editable module. Defaults to False, meaning a default task will use an `edit` implementation (i.e., only the interface for a module is defined).
        no_data: Whether to create the codebase without loading any of the data files. If the code loads a pretrained model, this will also be skipped.

    """
    config_path = str(Path(__file__).parent / f"discobench_configs/{task_name}.yaml")

    with open(config_path) as f:
        task_config = yaml.safe_load(f)

    train = not test

    task_domain = task_name.split("_")[0]

    MakeFiles(task_domain).make_files(task_config, train=train, use_base=use_base, no_data=no_data)
