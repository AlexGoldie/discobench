import pytest
import yaml

from discobench import create_config
from discobench.utils import get_domains


@pytest.mark.parametrize("domain", get_domains())
def test_example_config_valid(domain: str) -> None:
    """Ensure the example_config has correct default values, and expected keys."""
    expected_config = create_config(domain)

    example_config_path = f"discobench/example_configs/{domain}.yaml"
    with open(example_config_path) as f:
        example_config = yaml.safe_load(f)

    # Check all keys are there
    assert sorted(example_config.keys()) == sorted(expected_config.keys())

    for k, v in example_config.items():
        assert type(v) is type(expected_config[k])

    # Ensure default key values are used when appropriate
    assert example_config["template_backend"] == "default"
    assert example_config["source_path"] == f"task_src/{domain}"
    assert len(example_config["train_task_id"]) >= 1
    assert len(example_config["test_task_id"]) >= 1
    assert list(set(example_config["train_task_id"]) & set(example_config["test_task_id"])) == []
