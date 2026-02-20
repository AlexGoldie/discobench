import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

import discobench
from discobench.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.mark.parametrize("task_domain", ["OnPolicyRL", "OffPolicyRL", "abcdef"])
@pytest.mark.parametrize("test", [True, False])
@pytest.mark.parametrize("example", [True, False])
@pytest.mark.parametrize("use_base", [True, False])
@pytest.mark.parametrize("no_data", [True, False])
def test_create_task_cli(
    runner: CliRunner, task_domain: str, test: bool, example: bool, use_base: bool, no_data: bool
) -> None:
    """Test that create_task called correctly. We already test create_task separately, so we just want to check echos."""
    with patch("discobench.cli.create_task") as mock_create:

        def mock_behavior_error(
            task_domain: str, test: bool, config_path: str | None, example: bool, use_base: bool, no_data: bool
        ) -> None:
            if task_domain not in discobench.get_domains():
                raise ValueError("Invalid domain")

        mock_create.side_effect = mock_behavior_error

        args = ["create-task", "--task-domain", task_domain]
        if test:
            args.append("--test")
        if example:
            args.append("--example")
        if use_base:
            args.append("--use-base")
        if no_data:
            args.append("--no-data")

        results = runner.invoke(cli, args)

        if test and use_base:
            assert "--use-base has no effect" in results.output

        mock_create.assert_called_once_with(
            task_domain=task_domain, test=test, config_path=None, example=example, use_base=use_base, no_data=no_data
        )
        mode = "test" if test else "training"

        if task_domain != "abcdef":
            assert f"Successfully created {mode} task for domain: {task_domain}" in results.output
            assert results.exit_code == 0
        else:
            assert results.exit_code != 0


def test_get_domains(runner: CliRunner) -> None:
    """Test that get_domains echos in the expected way."""
    expected_domains = discobench.get_domains()

    results = runner.invoke(cli, "get-domains")

    assert results.exit_code == 0
    for domain in expected_domains:
        assert domain in results.output


def test_get_modules(runner: CliRunner) -> None:
    """Test that get_modules echos in the expected way."""
    expected_modules = discobench.get_modules()

    results = runner.invoke(cli, "get-modules")

    assert results.exit_code == 0
    for domain, modules in expected_modules.items():
        expected_line = f"{domain}: {', '.join(modules)}"
        assert expected_line in results.output


def test_create_config_cmd(runner: CliRunner, tmp_path: Path) -> None:
    """Test that create-config from path works."""
    task_domain = "OnPolicyRL"

    save_dir = str(tmp_path / "custom_configs")
    expected_file_path = os.path.join(save_dir, f"task_config_{task_domain}.yml")

    dummy_config = {"mocked_key": "mocked_value"}

    with patch("discobench.cli.create_config") as mock_create_config:
        mock_create_config.return_value = dummy_config

        results = runner.invoke(cli, ["create-config", "--task-domain", task_domain, "--save-dir", save_dir])

    assert results.exit_code == 0
    mock_create_config.assert_called_once_with(task_domain)

    assert os.path.exists(save_dir)
    assert os.path.isfile(expected_file_path)

    with open(expected_file_path) as f:
        saved_data = yaml.safe_load(f)

    assert saved_data == dummy_config
