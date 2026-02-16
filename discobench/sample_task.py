"""A function to sample a randomly generated task config. To prevent bias, this first uniformly samples a random task domain before sampling parameters for a task config for this task domain. Uses rejection sampling to remove invalid configs."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sample_task(
    p_edit: float,
    p_data: list[float],
    eval_type: str = "random",
    use_backends: bool = True,
    source_path: str = "task_src",
    max_attempts: int = 10,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Sample a random task using user defined variables.

    Args:
        p_edit: The probability a module is marked as editable. Must be between 0. and 1.
        p_data: A list of probabilities or weights for sampling. Supports either a list of 2 values, which must be [p_meta_train, p_meta_test], or a list of 3 values, which can be probabilities or weights [w_meta_train, w_meta_test, w_exclude].
        eval_type: What eval_type to use. Supports 'random', which will select a random eval_type, or one of ['performance', 'energy', 'time']. Defaults to 'random'.
        use_backends: Whether to only use the default backend, or randomly sample from the supported backend for each domain. Defaults to True.
        source_path: Where the task code should be saved after calling create_task() on the returned config.
        max_attempts: The max number of attempts supported for sampling a task from DiscoGen. Prevents the risk of inifinite or very long loops, if probabilities are set in such a way that tasks are valid tasks are hard to sample. Defaults to 10.
        seed: A random seed for reproducible task sampling. Defaults to None, in which case sampling will be non-deterministic.

    Returns:
        random_domain: The randomly sampled domain.
        new_config: A DiscoGen configuration dictionary.

    """
    _check_args(p_edit, p_data, eval_type, use_backends, source_path, max_attempts, seed)

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    discobench_path = Path(__file__).parent / "tasks"

    task_domains = [x for x in discobench_path.iterdir() if x.is_dir()]

    total_edit = 0
    total_train = 0
    total_test = 0
    attempts = 0

    while total_edit < 1 or total_train < 1 or total_test < 1 or attempts >= max_attempts:
        total_edit = 0
        total_train = 0
        total_test = 0
        attempts += 1

        random_domain = str(random.choice(task_domains))  # noqa: S311

        domain_yaml = discobench_path.joinpath(f"{random_domain}/task_config.yaml").read_text()
        domain_config = yaml.safe_load(domain_yaml)

        datasets = domain_config["train_task_id"]

        idx = [0, 1, 2]  # meta-train, meta-test, exclude

        dataset_idxes = np.random.choice(idx, size=len(datasets), p=p_data)

        new_config: dict[str, Any] = {"train_task_id": [], "test_task_id": []}

        if use_backends:
            new_config.update({"template_backend": "default"})

        else:
            templates_path = discobench_path.joinpath(f"{random_domain}/templates")
            backends = [x for x in templates_path.iterdir() if x.is_dir()]
            random_backend = random.choice(backends)  # noqa: S311
            new_config.update({"template_backend": random_backend})

        for i in range(len(datasets)):
            if dataset_idxes[i] == 0:
                new_config["train_task_id"].append(datasets[i])
                total_train += 1
            elif dataset_idxes[i] == 1:
                new_config["test_task_id"].append(datasets[i])
                total_test += 1

        for k in domain_config:
            if "change_" in k:
                rng = random.random()  # noqa: S311
                if rng < p_edit:
                    edit = True
                    total_edit += 1
                else:
                    edit = False

                new_config.update({k: edit})

        new_config["source_path"] = source_path

    return random_domain, new_config


def _check_args(
    p_edit: float,
    p_data: list[float],
    eval_type: str,
    use_backends: bool,
    source_path: str,
    max_attempts: int,
    seed: int | None,
) -> None:
    if not (len(p_data) == 2 or len(p_data) == 3):
        raise ValueError(
            "p_data must include probabilities for [meta_train, meta_test] or [meta_train, meta_test, exclude], meaning it must be length 2 or 3."
        )

    for p in p_data:
        if p < 0:
            raise ValueError("Each entry in p_data must be greater than or equal to 0.")

    if len(p_data) == 3:
        total_p_data = sum(p_data)
        p_data = [i / total_p_data for i in p_data]

    if p_edit <= 0 or p_edit > 1:
        raise ValueError("p_edit must be between 0 and 1.")

    if sum(p_data) > 1:
        raise ValueError("p_data must total to less than 1.")

    for p in p_data:
        if p >= 1:
            raise ValueError("Each entry in p_data must be less than 1.")

    if eval_type not in ["random", "performance", "energy", "time"]:
        raise ValueError("eval_type must be one of  ['random', 'performance', 'energy', 'time].")
