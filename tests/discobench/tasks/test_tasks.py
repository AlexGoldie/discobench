import os
from pathlib import Path

import pytest
import yaml

from discobench import create_config
from discobench.utils import get_domains


@pytest.mark.parametrize("domain", get_domains())
def test_task_datasets(domain: str) -> None:
    """Ensure every dataset has a description."""
    task_path = f"discobench/tasks/{domain}"
    dataset_path = f"{task_path}/datasets"

    file_list = []

    for _, (_, _, filenames) in enumerate(os.walk(dataset_path)):
        file_list.extend(filenames)
        break

    assert file_list == []  # There should only be directories here.

    task_config = create_config(domain)

    assert task_config["train_task_id"] == task_config["test_task_id"]

    found_in_datasets = set()
    task_data = set(task_config["train_task_id"])
    dataset_p = Path(dataset_path)
    for root, dirs, _ in os.walk(dataset_path, topdown=True):
        rel_path = Path(root).relative_to(dataset_p).as_posix()

        if rel_path == ".":
            continue

        if rel_path in task_data:
            found_in_datasets.add(rel_path)
            # We found a valid dataset, so we don't need to check subdirectories.
            dirs[:] = []
            assert os.path.exists(f"{root}/description.md")
        else:
            # If this is a leaf node and NOT in data_ids, it's an orphan and needs to be added to the config
            if not dirs:
                pytest.fail(f"Unauthorized leaf directory found: {rel_path}")

    # Check every dataset found is in the configs
    missing = task_data - found_in_datasets
    if missing:
        pytest.fail(f"Config IDs not found on disk: {missing}")


@pytest.mark.parametrize("domain", get_domains())
def test_task_utils(domain: str) -> None:
    """Ensure every dataset has the correct files in its setup, including that all files are copied over."""
    task_path = f"discobench/tasks/{domain}"
    utils_path = f"{task_path}/utils"

    expected_files = ["_reference.txt", "description.md", "requirements.txt", "task_information.yaml", "task_spec.yaml"]

    for file in expected_files:
        assert os.path.exists(f"{utils_path}/{file}")

    with open(f"{utils_path}/task_spec.yaml") as f:
        task_spec = yaml.safe_load(f)

    assert sorted(task_spec.keys()) == ["fixed_files", "module_files"]

    assert "main.py" in task_spec["fixed_files"]

    with open(f"{utils_path}/task_information.yaml") as f:
        task_information = yaml.safe_load(f)

    for file in task_spec["fixed_files"]:
        template_path = f"{task_path}/templates/default/{file}"

        if os.path.exists(template_path):
            pass
        else:
            datasets_dir = f"{task_path}/datasets"

            for dataset in create_config(domain)["train_task_id"]:
                dataset_file_path = os.path.join(datasets_dir, dataset, file)

                if not os.path.exists(dataset_file_path):
                    pytest.fail(
                        f"File {file} does not exist in either the main template, or the per-dataset templates."
                    )

    expected_information_keys = []

    for file in task_spec["module_files"]:
        base_path = f"{task_path}/templates/default/base/{file}"
        edit_path = f"{task_path}/templates/default/edit/{file}"

        assert os.path.exists(base_path)
        assert os.path.exists(edit_path)
        file_name_only = os.path.splitext(file)[0]
        expected_information_keys.append(f"{file_name_only}_prompt")

    assert sorted(task_information.keys()) == sorted(expected_information_keys)
